# modbus_driver.py
import time
import struct
from pymodbus.client import ModbusTcpClient


class TMModbus:
    def __init__(self, host: str, port: int = 502, unit_id: int = 1, timeout: float = 1.0):
        self.unit_id = unit_id
        self.client = ModbusTcpClient(host=host, port=port, timeout=timeout)

    def connect(self):
        if not self.client.connect():
            raise RuntimeError("Failed to connect to TM Modbus server")

    def close(self):
        self.client.close()

    # ---------------------------
    # pymodbus compatibility layer
    # ---------------------------
    def _call_with_unit(self, fn, *args, **kwargs):
        """
        Call a pymodbus client function using whatever unit keyword this install expects.
        Tries: slave=, unit=, device_id=, and finally no unit keyword.
        """
        # If caller already provided a unit-ish kwarg, just try it.
        try:
            return fn(*args, **kwargs)
        except TypeError:
            pass

        # Try common unit keyword variants
        for k in ("slave", "unit", "device_id"):
            try:
                return fn(*args, **kwargs, **{k: self.unit_id})
            except TypeError:
                continue

        # Last resort: call without any unit keyword
        return fn(*args, **kwargs)

    # -------- Register access --------

    def write_regs(self, addr: int, values: list[int]):
        # write_registers(address=..., values=...)
        rr = self._call_with_unit(self.client.write_registers, address=addr, values=values)
        if rr.isError():
            raise RuntimeError(rr)

    def write_reg(self, addr: int, value: int):
        # write_register(address=..., value=...)
        rr = self._call_with_unit(self.client.write_register, address=addr, value=value)
        if rr.isError():
            raise RuntimeError(rr)

    def read_regs(self, addr: int, count: int) -> list[int]:
        # read_holding_registers(address=..., count=...)
        rr = self._call_with_unit(self.client.read_holding_registers, address=addr, count=count)
        if rr.isError():
            raise RuntimeError(rr)
        return rr.registers

    # -------- Encoding helpers --------

    def float_to_regs(self, x: float, swap_words: bool = False) -> list[int]:
        """
        IEEE754 float32 → two 16-bit registers
        """
        b = struct.pack(">f", float(x))  # big-endian float32
        hi, lo = struct.unpack(">HH", b)
        return [lo, hi] if swap_words else [hi, lo]

    # -------- Mailbox command --------

    def send_command(
        self,
        cmd_code: int,
        seq: int,
        args_regs: list[int],
        CMD_CODE=9000,
        SEQ=9001,
        ACK=9002,
        ERR=9003,
        ARG_BASE=9010,
        ack_timeout: float = 3.0
    ):
        # Write command code + sequence
        self.write_regs(CMD_CODE, [cmd_code, seq])

        # Write arguments
        if args_regs:
            self.write_regs(ARG_BASE, args_regs)

        # Wait for ACK
        t0 = time.time()
        while time.time() - t0 < ack_timeout:
            ack = self.read_regs(ACK, 1)[0]
            err = self.read_regs(ERR, 1)[0]
            if ack == seq:
                return err
            time.sleep(0.05)

        raise TimeoutError("Timed out waiting for TM ACK")
