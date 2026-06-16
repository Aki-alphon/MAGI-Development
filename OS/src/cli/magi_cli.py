#!/usr/bin/env python3
"""
MAGI OS v2 — Command Line Interface
src/cli/magi_cli.py

Equivalent to the ros2 CLI tool. Inspect and control the MAGI system.

Usage:
    magi-cli topic list
    magi-cli topic echo /detections
    magi-cli topic hz   /sensors
    magi-cli param get  magi1 confidence_threshold
    magi-cli param set  magi1 confidence_threshold 0.6
    magi-cli param list magi1
    magi-cli node list
    magi-cli diag
    magi-cli record --topics /sensors /detections --output session.db
    magi-cli play   --file session.db --speed 2.0
    magi-cli info   --file session.db
    magi-cli tf     tree
"""

import sys, os, time, json, argparse, zmq, msgpack

sys.path.insert(0, "/opt/magi/src")

BUS_SUB  = "ipc:///tmp/magi/bus_sub.sock"
BUS_CTL  = "ipc:///tmp/magi/bus_ctl.sock"
PARAM_SV = "ipc:///tmp/magi/param_server.sock"


def _ctx():
    return zmq.Context()


def _ctl_req(cmd: dict) -> dict:
    ctx  = _ctx()
    sock = ctx.socket(zmq.REQ)
    sock.connect(BUS_CTL)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.send(msgpack.packb(cmd, use_bin_type=True))
    try:
        return msgpack.unpackb(sock.recv(), raw=False)
    except zmq.Again:
        return {"error": "Bus not responding"}


def _param_req(cmd: dict) -> dict:
    ctx  = _ctx()
    sock = ctx.socket(zmq.REQ)
    sock.connect(PARAM_SV)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.send(msgpack.packb(cmd, use_bin_type=True))
    try:
        return msgpack.unpackb(sock.recv(), raw=False)
    except zmq.Again:
        return {"error": "Param server not responding"}


# ─── topic subcommands ────────────────────────────────────────────────────────

def cmd_topic_list(args):
    resp = _ctl_req({"cmd": "topic_list"})
    topics = resp.get("topics", [])
    if not topics:
        print("No active topics (is message bus running?)")
        return
    print(f"Active topics ({len(topics)}):")
    for t in sorted(topics):
        print(f"  {t}")


def cmd_topic_echo(args):
    topic = args.topic
    ctx   = _ctx()
    sock  = ctx.socket(zmq.SUB)
    sock.connect(BUS_SUB)
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    print(f"Subscribing to {topic} — Ctrl+C to stop\n")
    count = 0
    try:
        while True:
            try:
                parts = sock.recv_multipart()
                data  = msgpack.unpackb(parts[1], raw=False)
                print(f"[{topic}] #{count}")
                print(json.dumps(data, indent=2, default=str))
                print()
                count += 1
                if args.count and count >= args.count:
                    break
            except zmq.Again:
                print(f"  (no message on {topic})")
    except KeyboardInterrupt:
        pass
    sock.close()


def cmd_topic_hz(args):
    topic = args.topic
    ctx   = _ctx()
    sock  = ctx.socket(zmq.SUB)
    sock.connect(BUS_SUB)
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
    sock.setsockopt(zmq.RCVTIMEO, 100)
    print(f"Measuring rate on {topic} — Ctrl+C to stop")
    times  = []
    window = 5.0
    try:
        while True:
            try:
                sock.recv_multipart()
                times.append(time.monotonic())
                # Keep only last `window` seconds
                cutoff = time.monotonic() - window
                times  = [t for t in times if t > cutoff]
                hz     = len(times) / window
                sys.stdout.write(f"\r  {hz:6.2f} Hz  ({len(times)} msgs / {window:.0f}s)   ")
                sys.stdout.flush()
            except zmq.Again:
                pass
    except KeyboardInterrupt:
        print()
    sock.close()


# ─── param subcommands ────────────────────────────────────────────────────────

def cmd_param_get(args):
    resp = _param_req({"cmd": "get", "node": args.node, "key": args.key})
    if "error" in resp:
        print(f"Error: {resp['error']}")
    elif resp.get("found"):
        print(f"{args.node}.{args.key} = {resp['value']!r}")
    else:
        print(f"Parameter '{args.key}' not found on node '{args.node}'")


def cmd_param_set(args):
    val = args.value
    # Auto-type conversion
    try:    val = int(val)
    except: 
        try:    val = float(val)
        except:
            if val.lower() == "true":  val = True
            elif val.lower() == "false": val = False

    resp = _param_req({"cmd": "set", "node": args.node, "key": args.key, "value": val})
    if resp.get("ok"):
        print(f"✅ Set {args.node}.{args.key} = {val!r}")
    else:
        print(f"❌ Failed: {resp}")


def cmd_param_list(args):
    resp = _param_req({"cmd": "list", "node": args.node})
    params = resp.get("params", {})
    print(f"Parameters for '{args.node}':")
    for k, v in sorted(params.items()):
        print(f"  {k:<35} = {v!r}")


# ─── diag subcommand ──────────────────────────────────────────────────────────

def cmd_diag(args):
    ctx  = _ctx()
    sock = ctx.socket(zmq.SUB)
    sock.connect(BUS_SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"/diagnostics")
    sock.setsockopt(zmq.RCVTIMEO, 500)

    nodes: dict = {}
    deadline = time.monotonic() + 6.0
    print("Collecting diagnostics (6s)...\n")
    while time.monotonic() < deadline:
        try:
            parts = sock.recv_multipart()
            d = msgpack.unpackb(parts[1], raw=False)
            nodes[d.get("node_id","?")] = d
        except zmq.Again:
            pass

    icons = {"OK": "✅", "WARN": "⚠️ ", "ERROR": "🔴", "STALE": "🟡"}
    print(f"{'NODE':<18} {'STATE':<14} {'STATUS':<8} MESSAGE")
    print("─" * 65)
    for nid, d in sorted(nodes.items()):
        st   = d.get("status","?")
        icon = icons.get(st, "❓")
        print(f"{nid:<18} {d.get('state','?'):<14} {icon} {st:<6}  {d.get('message','')}")
    sock.close()


# ─── record / play / info subcommands ────────────────────────────────────────

def cmd_record(args):
    from core.recorder import Recorder
    topics = args.topics if args.topics else ["/sensors", "/detections", "/scene", "/decision"]
    output = args.output or f"magi_{int(time.time())}.db"
    rec    = Recorder(topics=topics, output=output)
    rec.run()


def cmd_play(args):
    from core.recorder import Player
    player = Player(filepath=args.file, speed=args.speed or 1.0,
                    topics=args.topics or None)
    player.play()


def cmd_info(args):
    from core.recorder import Player
    info = Player.info(args.file)
    print(f"\nRecording: {args.file}")
    print(f"Metadata:")
    for k, v in info["metadata"].items():
        if "time" in k:
            try:
                v = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(float(v)))
            except: pass
        print(f"  {k:<15} = {v}")
    print(f"\nTopics:")
    for topic, cnt in info["topics"]:
        print(f"  {topic:<25} {cnt:>8} messages")


# ─── tf subcommand ────────────────────────────────────────────────────────────

def cmd_tf(args):
    from core.transforms import get_tf
    tf = get_tf()
    if args.subcmd == "tree":
        print(tf.tree())
    elif args.subcmd == "frames":
        for f in tf.all_frames():
            print(f" • {f}")


# ─── Main Parser ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="magi-cli", description="MAGI OS CLI")
    sub    = parser.add_subparsers(dest="cmd")

    # topic
    tp = sub.add_parser("topic")
    ts = tp.add_subparsers(dest="subcmd")
    ts.add_parser("list")
    te = ts.add_parser("echo");  te.add_argument("topic"); te.add_argument("--count", type=int, default=0)
    th = ts.add_parser("hz");    th.add_argument("topic")

    # param
    pp = sub.add_parser("param")
    ps = pp.add_subparsers(dest="subcmd")
    pg = ps.add_parser("get");   pg.add_argument("node"); pg.add_argument("key")
    pset = ps.add_parser("set"); pset.add_argument("node"); pset.add_argument("key"); pset.add_argument("value")
    pl = ps.add_parser("list");  pl.add_argument("node")

    # diag
    sub.add_parser("diag")

    # record
    rec = sub.add_parser("record")
    rec.add_argument("--topics", nargs="+")
    rec.add_argument("--output")

    # play
    ply = sub.add_parser("play")
    ply.add_argument("--file",   required=True)
    ply.add_argument("--speed",  type=float, default=1.0)
    ply.add_argument("--topics", nargs="+")

    # info
    inf = sub.add_parser("info");  inf.add_argument("--file", required=True)

    # tf
    tfp = sub.add_parser("tf")
    tfs = tfp.add_subparsers(dest="subcmd")
    tfs.add_parser("tree");  tfs.add_parser("frames")

    args = parser.parse_args()

    dispatch = {
        ("topic",  "list"):   cmd_topic_list,
        ("topic",  "echo"):   cmd_topic_echo,
        ("topic",  "hz"):     cmd_topic_hz,
        ("param",  "get"):    cmd_param_get,
        ("param",  "set"):    cmd_param_set,
        ("param",  "list"):   cmd_param_list,
        ("diag",   None):     cmd_diag,
        ("record", None):     cmd_record,
        ("play",   None):     cmd_play,
        ("info",   None):     cmd_info,
        ("tf",     "tree"):   cmd_tf,
        ("tf",     "frames"): cmd_tf,
    }

    key = (args.cmd, getattr(args, "subcmd", None))
    fn  = dispatch.get(key)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
