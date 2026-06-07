from __future__ import annotations

import socket
import threading
import time
import webbrowser

from waitress import serve

from web_app.app import create_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _open_browser(url: str) -> None:
    time.sleep(0.9)
    webbrowser.open(url)


def main() -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    print(f"PerfectPixel Web 正在运行：{url}")
    print("关闭此窗口即可停止应用。")
    serve(create_app(), host="127.0.0.1", port=port, threads=4)


if __name__ == "__main__":
    main()
