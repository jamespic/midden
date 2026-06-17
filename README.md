Midden
======

Midden is a tool for dumping and analysing heaps from Python programs.

It takes a "dump first, ask questions later" approach, making it easy to grab a heap dump from a running
application, then analyse it offline later in its UI, potentially on a different machine entirely.

Installing
----------

If you just want to grab a heap dump, you can install everything you need with:

```
pip install midden
# Or if you're using uv
uv add midden
```

If you want to analyse the data, you'll need the extra UI dependencies, which you can install with

```
pip install midden[ui]
# Or if you're using uv
uv add midden[ui]
```

Grabbing a Heap Dump
--------------------

Grabbing a heap dump can be as simple as:

```
# Assuming pid we want to grab heap from is pid 12345
midden-inject 12345 --output-file /tmp/dump.jsonl
```

On Python 3.14 and newer, this will use `sys.remote_exec`, which means no extra dependencies.
You will need appropriate permissions [as documented here](https://docs.python.org/3/howto/remote_debugging.html#permission-requirements).
Roughly speaking, you either need to be root/administrator, or be running on Linux with ptrace protection disabled.

On Python 3.10 to 3.13, injection is done with gdb, which means gdb needs to be installed.
Gdb-based injection is only tested on Linux.

Analysing a Heap Dump
---------------------

You can run the analysis UI with

```
midden-ui
```

This will start a web application, and pop up a web browser pointing at the analysis application. Upload a heap dump
generated with `midden-inject` to get started.
