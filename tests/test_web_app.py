from pathlib import Path
import base64
import io
import json
import unittest
import zipfile

import cv2
import numpy as np

from web_app.app import create_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")

    def test_process_image(self):
        image_path = Path(__file__).parents[1] / "images" / "girl.jpg"
        with image_path.open("rb") as image:
            response = self.client.post(
                "/api/process",
                data={
                    "image": (image, "girl.jpg"),
                    "sample_method": "center",
                    "backend": "auto",
                    "auto_grid": "true",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["result"].startswith("data:image/png;base64,"))
        self.assertEqual(response.json["diagnostics"]["output_width"], 24)
        self.assertLessEqual(len(response.json["diagnostics"]["grid_candidates"]), 3)
        self.assertIsNotNone(response.json["diagnostics"]["grid_confidence"])

    def test_process_image_with_numpy_backend(self):
        image_path = Path(__file__).parents[1] / "images" / "avatar.png"
        with image_path.open("rb") as image:
            response = self.client.post(
                "/api/process",
                data={
                    "image": (image, "avatar.png"),
                    "sample_method": "median",
                    "backend": "numpy",
                    "auto_grid": "false",
                    "grid_width": "32",
                    "grid_height": "32",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["diagnostics"]["backend"], "NumPy 轻量后端")
        self.assertEqual(response.json["diagnostics"]["grid_candidates"], [])
        self.assertIsNone(response.json["diagnostics"]["grid_confidence"])

    def test_process_image_with_adaptive_sampling(self):
        image_path = Path(__file__).parents[1] / "images" / "girl.jpg"
        with image_path.open("rb") as image:
            response = self.client.post(
                "/api/process",
                data={
                    "image": (image, "girl.jpg"),
                    "sample_method": "adaptive",
                    "backend": "auto",
                    "auto_grid": "false",
                    "grid_width": "24",
                    "grid_height": "24",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["diagnostics"]["sample_method"], "adaptive")

    def test_manual_grid_supports_long_strip_over_512_cells(self):
        rgb = np.zeros((1200, 80, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(success)

        response = self.client.post(
            "/api/process",
            data={
                "image": (io.BytesIO(encoded.tobytes()), "strip.png"),
                "sample_method": "center",
                "backend": "auto",
                "auto_grid": "false",
                "grid_width": "40",
                "grid_height": "600",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["diagnostics"]["output_width"], 40)
        self.assertEqual(response.json["diagnostics"]["output_height"], 600)

    def test_manual_grid_cannot_exceed_source_dimensions(self):
        response = self.client.post(
            "/api/process",
            data={
                "image": (io.BytesIO(self._transparent_png()), "small.png"),
                "sample_method": "center",
                "backend": "auto",
                "auto_grid": "false",
                "grid_width": "32",
                "grid_height": "128",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("不能超过原图尺寸", response.json["error"])

    def test_transparent_png_preserves_alpha(self):
        transparent_png = self._transparent_png()

        response = self.client.post(
            "/api/process",
            data={
                "image": (io.BytesIO(transparent_png), "transparent.png"),
                "sample_method": "adaptive",
                "backend": "auto",
                "auto_grid": "false",
                "grid_width": "16",
                "grid_height": "16",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = base64.b64decode(response.json["result"].split(",", 1)[1])
        output = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(output.shape[2], 4)
        self.assertEqual(int(output[..., 3].min()), 0)
        self.assertEqual(int(output[..., 3].max()), 255)
        self.assertTrue(response.json["diagnostics"]["has_alpha"])

    def test_process_batch_returns_zip_and_report(self):
        root = Path(__file__).parents[1]
        files = []
        for name in ("girl.jpg", "avatar.png"):
            files.append((io.BytesIO((root / "images" / name).read_bytes()), f"sample-folder/{name}"))

        response = self.client.post(
            "/api/process-batch",
            data={
                "images": files,
                "sample_method": "center",
                "backend": "auto",
                "auto_grid": "false",
                "grid_width": "32",
                "grid_height": "32",
                "export_scale": "2",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        self.assertEqual(response.headers["X-Batch-Success"], "2")
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            names = archive.namelist()
            self.assertIn("results/sample-folder/girl_perfect.png", names)
            self.assertIn("results/sample-folder/avatar_perfect.png", names)
            exported = cv2.imdecode(
                np.frombuffer(archive.read("results/sample-folder/girl_perfect.png"), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertEqual(exported.shape[:2], (64, 64))
            report = json.loads(archive.read("perfect-pixel-report.json"))
            self.assertEqual(report["export_scale"], 2)
            self.assertEqual(report["failed"], 0)

    def test_batch_transparent_png_preserves_alpha(self):
        response = self.client.post(
            "/api/process-batch",
            data={
                "images": [(io.BytesIO(self._transparent_png()), "transparent.png")],
                "sample_method": "adaptive",
                "backend": "auto",
                "auto_grid": "false",
                "grid_width": "16",
                "grid_height": "16",
                "export_scale": "2",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            output = cv2.imdecode(
                np.frombuffer(archive.read("results/transparent_perfect.png"), dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(output.shape, (32, 32, 4))
            self.assertEqual(int(output[..., 3].min()), 0)

    @staticmethod
    def _transparent_png():
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[16:48, 16:48, :3] = (40, 90, 220)
        rgba[16:48, 16:48, 3] = 255
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        success, encoded = cv2.imencode(".png", bgra)
        assert success
        return encoded.tobytes()


if __name__ == "__main__":
    unittest.main()
