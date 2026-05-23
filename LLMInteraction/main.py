# from logger import log_action
import os, io, time, sys, wave, uuid
import numpy as np
import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from logger import log_action
from modbus_driver import TMModbus
from logger import log_action, log_chat
def load_tools():
    path = os.path.join("functions", "robot_functions.json")
    with open(path, "r") as f:
        return json.load(f)

TOOLS = load_tools()

def load_locations():
    path = os.path.join("data", "locations.json")
    with open(path, "r") as f:
        return json.load(f)

LOCATIONS = load_locations()

def resolve_pose(name: str):
    try:
        pose = LOCATIONS["poses"][name]
        return pose
    except KeyError:
        raise ValueError(f"Unknown location '{name}' in locations.json")
def pose_to_arg_regs(pose: dict) -> list[int]:
    # x,y,z,rx,ry,rz as float32 each -> 12 registers total
    regs = []
    regs += tm.float_to_regs(pose["x"], swap_words=TM_SWAP_WORDS)
    regs += tm.float_to_regs(pose["y"], swap_words=TM_SWAP_WORDS)
    regs += tm.float_to_regs(pose["z"], swap_words=TM_SWAP_WORDS)
    regs += tm.float_to_regs(pose["rx"], swap_words=TM_SWAP_WORDS)
    regs += tm.float_to_regs(pose["ry"], swap_words=TM_SWAP_WORDS)
    regs += tm.float_to_regs(pose["rz"], swap_words=TM_SWAP_WORDS)
    return regs

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TM_HOST = os.getenv("TM_HOST", "172.31.1.22")
TM_PORT = int(os.getenv("TM_PORT", "502"))
TM_UNIT_ID = int(os.getenv("TM_UNIT_ID", "1"))
TM_SWAP_WORDS = os.getenv("TM_SWAP_WORDS", "1") == "0"

tm = TMModbus(host=TM_HOST, port=TM_PORT, unit_id=TM_UNIT_ID)
tm.connect()

seq = 0  # global sequence counter

STT_MODEL  = os.getenv("OPENAI_STT_MODEL",  "gpt-4o-transcribe")  # or whisper-1
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TTS_MODEL  = os.getenv("OPENAI_TTS_MODEL",  "gpt-4o-mini-tts")
TTS_VOICE  = os.getenv("OPENAI_TTS_VOICE",  "alloy")

FS = 16000       # 16 kHz mono is a good default for STT
SECONDS = 4      # seconds per utterance; change as you like

def record_wav(path: str, seconds: int = SECONDS, fs: int = FS):
    print(f"[rec] Recording {seconds}s… (Ctrl+C to stop)")
    audio = sd.rec(int(seconds*fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, fs, subtype="PCM_16")
    print(f"[rec] Saved {path}")

def play_audio(path: str):
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def transcribe_wav(path: str) -> str:
    # OpenAI Audio Transcriptions API (Python SDK)
    # Docs: Speech-to-Text guide & API reference
    # Returns plain text. :contentReference[oaicite:1]{index=1}
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f
        )
    text = resp.text if hasattr(resp, "text") else getattr(resp, "text", "")
    print(f"[stt] {text}")
    return text

# def chat_llm(user_text: str) -> str:
#     # Chat Completions API (Python SDK) :contentReference[oaicite:2]{index=2}
#     msgs = [
#         {"role": "system",
#          "content": ("You are the Disassembly Assistant for human–robot collaboration. "
#                      "Respond with: 1) next safe step, 2) hazards check, "
#                      "3) concise robot function JSON when appropriate. "
#                      "Keep answers < 59 words; never suggest unsafe actions.")},
#         {"role": "user", "content": user_text}
#     ]
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=msgs
#     )
#     reply = resp.choices[0].message.content.strip()
#     print(f"[llm] {reply}")
#     return reply

# def chat_llm(user_text: str) -> str:
#     msgs = [
#         {
#             "role": "system",
#             "content": (
#                 "You are the Disassembly Assistant for human–robot collaboration.\n"
#                 "When appropriate, call tools instead of just replying in text.\n"
#                 "Only call tools that correspond to safe, realistic actions."
#             ),
#         },
#         {"role": "user", "content": user_text},
#     ]
#
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=msgs,
#         tools=TOOLS,
#         tool_choice="auto",
#     )
#
#     msg = resp.choices[0].message
#
#     # If it chose a tool, print it (later you’ll execute it).
#     if msg.tool_calls:
#         print("[tool] model requested tools:")
#         for call in msg.tool_calls:
#             fn_name = call.function.name
#             args = json.loads(call.function.arguments or "{}")
#             # print(f"   -> {fn_name}({args})")
#             # # TODO: map fn_name + args to your robot / logger
#             print(f"[tool-call] {fn_name} -> {args}")
#
#             # TODO: Map calls to real robot functions later
#         # You can still return a natural-language summary:
#         return "I’ve prepared a robot action. Check the tool logs."
#     else:
#         reply = msg.content.strip()
#         print(f"[llm] {reply}")
#         return reply

def chat_llm(user_text: str) -> str:
    # Log user message
    log_chat(role="user", content=user_text)
    
    msgs = [
        {
            "role": "system",
            "content": (
                "You are the Disassembly Assistant for human–robot collaboration.\n"
                "When appropriate, call tools instead of just replying in text.\n"
                "Only call tools that correspond to safe, realistic actions.\n"
                "After calling tools, the system will summarise what you did for the human."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=msgs,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = resp.choices[0].message

    # If it chose a tool, handle the tool calls
    if msg.tool_calls:
        print("[tool] model requested tools:")

        # We'll collect human-friendly descriptions here
        descriptions: list[str] = []

        for call in msg.tool_calls:
            fn_name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            print(f"[tool-call] {fn_name} -> {args}")

            # ---- TOOL HANDLER SECTION ----

            # if fn_name == "move_robot":
            #     loc_name = args["location_name"]
            #     pose = resolve_pose(loc_name)
            #
            #     seq = (seq + 1) & 0xFFFF  # keep seq global
            #     arg_regs = build_move_pose_args(tm, pose)
            #
            #     err = tm.send_command(cmd_code=CMD_MOVE_POSE, seq=seq, args_regs=arg_regs)
            #     if err == 0:
            #         descriptions.append(f"Okay — moving to '{loc_name}'.")
            #     else:
            #         descriptions.append(f"I tried moving to '{loc_name}', but the robot reported error code {err}.")

            if fn_name == "move_robot":
                loc_name = args["location_name"]
                speed = args.get("speed", 0.2)

                pose = resolve_pose(loc_name)
                print(f"[robot] Move to {loc_name}: pose={pose}, speed={speed}")
                global seq
                seq = (seq + 1) & 0xFFFF

                CMD_MOVE_POSE = 10  # choose your own command codes
                arg_regs = pose_to_arg_regs(pose)

                err = tm.send_command(cmd_code=CMD_MOVE_POSE, seq=seq, args_regs=arg_regs)
                if err == 0:
                    descriptions.append(f"Okay — moving to '{loc_name}'.")
                else:
                    descriptions.append(f"I tried moving to '{loc_name}', but the robot returned error code {err}.")

                # if err != 0:
                #     descriptions.append(f"However, the robot reported error code {err}.")

                log_action(
                    tool_name="move_robot",
                    args=args,
                    resolved_pose=pose,
                    message=f"Move robot to {loc_name} at speed {speed}"
                )

                # descriptions.append(
                #     f"I'm moving the robot to the '{loc_name}' position at speed {speed:.2f}."
                # )

            elif fn_name == "unscrew_fastener":
                fastener_id = args["fastener_id"]
                torque_nm = args.get("torque_nm", 1.0)
                direction = args["direction"]

                print(f"[robot] Unscrew {fastener_id}: torque={torque_nm}, dir={direction}")
                # TODO: robot.unscrew(fastener_id, torque_nm, direction)

                log_action(
                    tool_name="unscrew_fastener",
                    args=args,
                    resolved_pose=None,
                    message=f"Unscrew fastener {fastener_id}"
                )

                action = "loosening" if direction == "ccw" else "tightening"
                descriptions.append(
                    f"I'm {action} fastener '{fastener_id}' with approximately {torque_nm:.2f} newton-metres of torque."
                )

            elif fn_name == "grip_part":
                part_id = args["part_id"]
                grip_mode = args.get("grip_mode", "pinch")

                print(f"[robot] Grip {part_id} using mode={grip_mode}")
                # TODO: robot.grip(part_id, grip_mode)

                log_action(
                    tool_name="grip_part",
                    args=args,
                    message=f"Gripping part {part_id} using {grip_mode} mode"
                )

                descriptions.append(
                    f"I'm gripping part '{part_id}' using a {grip_mode} grip."
                )

            elif fn_name == "place_part":
                part_id = args["part_id"]
                location_id = args["location_id"]
                pose = resolve_pose(location_id)

                print(f"[robot] Place {part_id} at {location_id}: pose={pose}")
                # TODO: robot.place_part(part_id, pose)

                log_action(
                    tool_name="place_part",
                    args=args,
                    resolved_pose=pose,
                    message=f"Placing {part_id} at {location_id}"
                )

                descriptions.append(
                    f"I'm placing part '{part_id}' into location '{location_id}'."
                )

            elif fn_name == "inspect_with_camera":
                target_id = args["target_id"]
                view = args.get("view", "overall")

                print(f"[robot] Inspect {target_id} with view '{view}'")
                # TODO: robot.inspect(target_id, view)

                log_action(
                    tool_name="inspect_with_camera",
                    args=args,
                    message=f"Inspect {target_id} (view={view})"
                )

                descriptions.append(
                    f"I'm inspecting '{target_id}' with the camera using a {view} view."
                )

            elif fn_name == "request_human_assistance":
                reason = args["reason"]
                suggestion = args.get("suggested_action", "")

                print(f"[assist] Human assistance required: {reason}. Suggest: {suggestion}")
                # TODO: request_human_assistance(reason, suggestion)

                log_action(
                    tool_name="request_human_assistance",
                    args=args,
                    message=f"Request human assistance: {reason}"
                )

                base = f"I need your help: {reason}."
                if suggestion:
                    base += f" Suggested action: {suggestion}."
                descriptions.append(base)

            elif fn_name == "detect_screws":
                component_id = args.get("component_id")
                region = args.get("region")

                print(f"[robot] Detect screws: component={component_id}, region={region}")
                # TODO: robot.detect_screws(component_id, region)

                log_action(
                    tool_name="detect_screws",
                    args=args,
                    message=f"Detect screws (component={component_id}, region={region})"
                )

                if component_id:
                    descriptions.append(
                        f"I'm detecting screws on component '{component_id}'."
                    )
                elif region:
                    descriptions.append(
                        f"I'm detecting screws in the '{region}' region."
                    )
                else:
                    descriptions.append(
                        "I'm detecting screws in the workspace."
                    )

            elif fn_name == "emergency_stop":
                reason = args.get("reason", "Emergency stop triggered")

                print(f"[emergency] Emergency stop: {reason}")
                # TODO: robot.emergency_stop(reason)

                log_action(
                    tool_name="emergency_stop",
                    args=args,
                    message=f"Emergency stop: {reason}"
                )

                descriptions.append(
                    f"Emergency stop activated. {reason}"
                )

            elif fn_name == "go_to_next_task":
                task_id = args.get("task_id")

                print(f"[task] Go to next task: task_id={task_id}")
                # TODO: go_to_next_task(task_id)

                log_action(
                    tool_name="go_to_next_task",
                    args=args,
                    message=f"Go to next task (task_id={task_id})"
                )

                if task_id:
                    descriptions.append(
                        f"Advancing to task '{task_id}'."
                    )
                else:
                    descriptions.append(
                        "Advancing to the next task in the disassembly plan."
                    )

            elif fn_name == "stop_motion":
                mode = args.get("mode", "soft_stop")

                print(f"[robot] Stop motion: mode={mode}")
                # TODO: robot.stop_motion(mode)

                log_action(
                    tool_name="stop_motion",
                    args=args,
                    message=f"Stop motion (mode={mode})"
                )

                mode_desc = "soft stop" if mode == "soft_stop" else "hold position"
                descriptions.append(
                    f"I'm stopping robot motion with a {mode_desc}."
                )

            elif fn_name == "resume_motion":
                from_state = args.get("from_state")

                print(f"[robot] Resume motion: from_state={from_state}")
                # TODO: robot.resume_motion(from_state)

                log_action(
                    tool_name="resume_motion",
                    args=args,
                    message=f"Resume motion (from_state={from_state})"
                )

                if from_state:
                    state_desc = "last motion" if from_state == "last_motion" else "safe pose"
                    descriptions.append(
                        f"I'm resuming robot motion from the {state_desc}."
                    )
                else:
                    descriptions.append(
                        "I'm resuming robot motion."
                    )

            elif fn_name == "start_timer":
                timer_id = args["timer_id"]
                label = args.get("label", "")

                print(f"[timer] Start timer: timer_id={timer_id}, label={label}")
                # TODO: start_timer(timer_id, label)

                log_action(
                    tool_name="start_timer",
                    args=args,
                    message=f"Start timer {timer_id}" + (f" ({label})" if label else "")
                )

                if label:
                    descriptions.append(
                        f"I'm starting timer '{timer_id}' for '{label}'."
                    )
                else:
                    descriptions.append(
                        f"I'm starting timer '{timer_id}'."
                    )

            elif fn_name == "stop_timer":
                timer_id = args["timer_id"]

                print(f"[timer] Stop timer: timer_id={timer_id}")
                # TODO: stop_timer(timer_id)

                log_action(
                    tool_name="stop_timer",
                    args=args,
                    message=f"Stop timer {timer_id}"
                )

                descriptions.append(
                    f"I'm stopping timer '{timer_id}' and recording the elapsed time."
                )

            else:
                print(f"[warning] Unknown tool call: {fn_name}")
                descriptions.append(
                    "I've prepared an internal robot action that I can't fully describe yet."
                )

            # ---- END TOOL HANDLER SECTION ----

        # Build a friendly spoken summary
        if len(descriptions) == 1:
            reply = descriptions[0]
        else:
            # Join multiple actions nicely
            # e.g. "First, ..., Then, ..."
            first = descriptions[0]
            rest = " ".join(
                "Then, " + d[0].lower() + d[1:] if d.endswith(".") else "Then, " + d
                for d in descriptions[1:]
            )
            reply = first + " " + rest

        print(f"[llm-summary] {reply}")
        # Log LLM response
        log_chat(role="assistant", content=reply)
        return reply

    # Normal text response (no tools used)
    else:
        reply = msg.content.strip()
        print(f"[llm] {reply}")
        # Log LLM response
        log_chat(role="assistant", content=reply)
        return reply


def tts_to_wav(text: str, path: str):
    # Text-to-Speech using OpenAI Python SDK v1.x
    # Uses response_format="wav" and streams directly to a file.
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",   # <-- this is the correct keyword
    )

    # This method is provided by the SDK to save the audio file
    speech.stream_to_file(path)

    print(f"[tts] Saved {path}")




def run_turn():
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    os.makedirs("runs", exist_ok=True)
    inp = f"runs/{run_id}_in.wav"
    out = f"runs/{run_id}_out.wav"

    record_wav(inp)
    transcript = transcribe_wav(inp)
    if not transcript:
        print("[warn] Empty transcript; try speaking closer to the mic.")
        return

    reply = chat_llm(transcript)
    tts_to_wav(reply, out)
    play_audio(out)

def main():
    print("Push-to-talk: press ENTER to record, 'q' + ENTER to quit.")
    while True:
        try:
            cmd = input("> ").strip().lower()
        except KeyboardInterrupt:
            print("\nBye"); break

        if cmd == "q":
            print("Bye"); break
        run_turn()

if __name__ == "__main__":
    main()

