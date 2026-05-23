import os
from dotenv import load_dotenv
from modbus_driver import TMModbus


load_dotenv()
tm = TMModbus(
    host=os.getenv("TM_HOST", "172.31.1.22"),
    port=int(os.getenv("TM_PORT", "502")),
    unit_id=int(os.getenv("TM_UNIT_ID", "1"))
)
tm.connect()
print("Connected OK")

# Try reading the mailbox ACK/ERR
ack = tm.read_regs(9002, 1)[0]
err = tm.read_regs(9003, 1)[0]
print("ACK:", ack, "ERR:", err)

# Try writing a dummy command (TMflow must be running to ack it)
tm.write_regs(9000, [99, 101])   # cmd=99 seq=1
tm.write_regs(9010, [0]*12)

print("Wrote dummy command. If TMflow is set, it should ack.")
tm.close()
