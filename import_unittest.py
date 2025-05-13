import unittest
from flask import Flask
from app import app

# test_app.py

class TestApp(unittest.TestCase):
    def setUp(self):
        # Set up the Flask test client
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        # Test the index route
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.data)  # Check if HTML content is returned

    def test_video_feed_route(self):
        # Test the video feed route
        response = self.app.get('/video_feed')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'multipart/x-mixed-replace; boundary=frame')

    def test_predict_route_no_prediction(self):
        # Test the predict route when no prediction is available
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"error": "No prediction available"', response.data)

if __name__ == '__main__':
    unittest.main()