import signal
import sys
from time import sleep

data = [
    "Test Data Item",
    {"Test Data Key": "Test Data Value"},
    (1,),
    {2.0},
    frozenset([b"3"]),
]

signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
try:
    print("Started!!!")
    sys.stdout.flush()  # Ensure the output is flushed so the test can detect it
    while True:
        sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
