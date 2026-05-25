"""new-prototype: scaffolds a new instrument prototype."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent

_PROTO_DIR = _REPO_ROOT / "prototypes"

_PATCH_PY = '''\
"""{name}: <describe what this patch does>."""

from __future__ import annotations

from base.output.overlay import OverlayConfig
from base.signals.bus import OutputBus, SignalBus


class {class_name}:
    """<one-line description>"""

    overlay = OverlayConfig()

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        pass  # reinitialise state on seek/restart

    def shutdown(self, outputs: OutputBus) -> None:
        pass  # send cleanup messages to PD before exit (e.g. silence notes)

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        # Called every loop iteration. Read signals, send messages to PD:
        #   outputs.send("message_name", value)
        # See PROTOTYPING.md for all available signals.
        pass
'''

_INIT_PY = '''\
from base.patches import register_patch
from .patch import {class_name}

register_patch("{name}", {class_name})

__all__ = ["{class_name}"]
'''

_PD_PATCH = '''\
#N canvas 50 80 640 500 12;
#X text 30 20 === {name} ===;
#X text 30 50 Add your synthesis below. Connect audio to [dac~]., f 70;
#X text 30 80 See PROTOTYPING.md for how to use the Python signals.;
#X obj 30 160 netreceive 9001;
#X text 295 160 <- receives FUDI messages from Python (TCP 9001);
#X obj 30 195 route my_message;
#X text 265 195 <- add your message names here;
#X text 30 260 Add synthesis objects here...;
#X obj 30 430 dac~;
#X connect 3 0 5 0;
'''


def _to_class_name(name: str) -> str:
    parts = re.split(r"[_\-]", name)
    return "".join(p[0].upper() + p[1:] for p in parts if p) + "Patch"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold a new eye-synth instrument prototype.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: uv run new-prototype TNC_v2",
    )
    parser.add_argument("name", help="Patch name, e.g. TNC_v2 or MyPatch_v1")
    args = parser.parse_args()

    name: str = args.name
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]*", name):
        sys.exit("error: name must start with a letter and contain only letters, digits, _ or -")

    proto_dir = _PROTO_DIR / name
    pd_file = proto_dir / f"{name}.pd"

    if proto_dir.exists():
        sys.exit(f"error: {proto_dir} already exists")

    class_name = _to_class_name(name)

    proto_dir.mkdir(parents=True)
    (proto_dir / "patch.py").write_text(_PATCH_PY.format(name=name, class_name=class_name))
    (proto_dir / "__init__.py").write_text(_INIT_PY.format(name=name, class_name=class_name))
    pd_file.write_text(_PD_PATCH.format(name=name))

    print(f"Created {name}")
    print(f"  Python: prototypes/{name}/patch.py")
    print(f"  PD:     prototypes/{name}/{name}.pd")
    print()
    print("Next steps:")
    print(f"  1. Edit prototypes/{name}/patch.py")
    print(f"  2. Open prototypes/{name}/{name}.pd in Pure Data")
    print(f"  3. uv run pupil-tracker --patch {name}")
    print(f"     uv run pupil-player recordings/000 --patch {name}")


if __name__ == "__main__":
    main()
