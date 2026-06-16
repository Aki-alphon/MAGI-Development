"""
MAGI OS v2 — SQLite Message Recorder
src/core/recorder.py

Records all topic messages to SQLite — lightweight rosbag equivalent.
Supports selective recording, playback at N× speed, and inspection.

Usage (from magi-cli):
    magi-cli record --topics /sensors /detections --output session.db
    magi-cli play   --file session.db --speed 2.0
    magi-cli info   --file session.db
"""

import os, sys, time, signal, sqlite3, threading, zmq, msgpack
sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

from common.logger import get_logger

log = get_logger("recorder")

BUS_SUB = "tcp://localhost:5556"
BUS_PUB = "tcp://localhost:5555"

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    topic     TEXT    NOT NULL,
    stamp     REAL    NOT NULL,
    wall_ts   REAL    NOT NULL,
    msg_type  TEXT,
    data      BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topic ON messages(topic);
CREATE INDEX IF NOT EXISTS idx_stamp ON messages(stamp);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Recorder:
    """
    Subscribes to specified topics and writes every message to SQLite.
    """

    def __init__(self, topics: list[str], output: str, max_mb: int = 500):
        self._topics   = topics
        self._output   = output
        self._max_bytes = max_mb * 1024 * 1024
        self._running   = True
        self._count     = 0
        self._ctx       = zmq.Context()

        # SQLite setup
        self._db = sqlite3.connect(output, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)

        # Write session metadata
        self._db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("start_time", str(time.time())))
        self._db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("topics", ",".join(topics)))
        self._db.commit()

        # ZMQ subscriber
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(BUS_SUB)
        for t in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, t.encode())
        self._sock.setsockopt(zmq.RCVTIMEO, 200)

        # Batch writer thread
        self._queue   = []
        self._q_lock  = threading.Lock()
        self._writer  = threading.Thread(target=self._batch_write, daemon=True)
        self._writer.start()

        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT,  self._shutdown)

        log.info(f"Recording topics: {topics} → {output}")

    def _batch_write(self):
        """Write queued messages to DB every 500ms."""
        while self._running:
            time.sleep(0.5)
            with self._q_lock:
                if not self._queue:
                    continue
                batch = self._queue[:]
                self._queue.clear()
            try:
                self._db.executemany(
                    "INSERT INTO messages (topic, stamp, wall_ts, msg_type, data) VALUES (?,?,?,?,?)",
                    batch
                )
                self._db.commit()
            except Exception as e:
                log.error(f"DB write error: {e}")

    def run(self):
        log.info("Recorder running...")
        t_start = time.time()
        while self._running:
            # Check size limit
            if os.path.getsize(self._output) > self._max_bytes:
                log.warning(f"Recording limit reached ({self._max_bytes//1024//1024} MB). Stopping.")
                break
            try:
                parts  = self._sock.recv_multipart()
                topic  = parts[0].decode()
                raw    = parts[1]
                now    = time.time()
                stamp  = now - t_start   # Relative timestamp for playback
                msg_type = ""
                try:
                    d = msgpack.unpackb(raw, raw=False)
                    msg_type = d.get("__type__", "")
                except Exception:
                    pass

                with self._q_lock:
                    self._queue.append((topic, stamp, now, msg_type, raw))
                self._count += 1

                if self._count % 1000 == 0:
                    sz = os.path.getsize(self._output) / 1024 / 1024
                    log.info(f"Recorded {self._count} messages | {sz:.1f} MB")

            except zmq.Again:
                pass

        self._flush_and_close()

    def _flush_and_close(self):
        with self._q_lock:
            if self._queue:
                self._db.executemany(
                    "INSERT INTO messages (topic, stamp, wall_ts, msg_type, data) VALUES (?,?,?,?,?)",
                    self._queue
                )
        self._db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("end_time",   str(time.time())))
        self._db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("msg_count",  str(self._count)))
        self._db.commit()
        self._db.close()
        self._sock.close()
        log.info(f"Recording saved: {self._count} messages → {self._output}")

    def _shutdown(self, sig, frame):
        self._running = False


class Player:
    """Replay a recorded session to the MAGI bus at N× speed."""

    def __init__(self, filepath: str, speed: float = 1.0, topics: list[str] = None):
        self._db     = sqlite3.connect(filepath)
        self._speed  = max(0.1, speed)
        self._topics = topics
        self._ctx    = zmq.Context()
        self._pub    = self._ctx.socket(zmq.PUB)
        self._pub.connect(BUS_PUB)
        time.sleep(0.2)
        log.info(f"Player ready: {filepath} at {speed}× speed")

    def play(self):
        where = ""
        if self._topics:
            ph = ",".join("?" * len(self._topics))
            where = f"WHERE topic IN ({ph})"

        rows = self._db.execute(
            f"SELECT topic, stamp, data FROM messages {where} ORDER BY stamp",
            self._topics or ()
        ).fetchall()

        if not rows:
            log.warning("No messages to play.")
            return

        log.info(f"Playing {len(rows)} messages...")
        t0      = time.monotonic()
        t_start = rows[0][1]

        for topic, stamp, data in rows:
            target = t0 + (stamp - t_start) / self._speed
            delay  = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._pub.send_multipart([topic.encode(), data])

        log.info("Playback complete.")
        self._pub.close()

    @staticmethod
    def info(filepath: str) -> dict:
        db   = sqlite3.connect(filepath)
        meta = dict(db.execute("SELECT key, value FROM metadata").fetchall())
        topics = db.execute(
            "SELECT topic, COUNT(*) as cnt FROM messages GROUP BY topic ORDER BY cnt DESC"
        ).fetchall()
        db.close()
        return {"metadata": meta, "topics": topics}
