from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print("=== POST REQUEST ===")
        print(f"Path: {self.path}")
        print("Headers:")
        for k, v in self.headers.items():
            print(f"  {k}: {v}")
        
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            print("Body:")
            print(body.decode('utf-8')[:500])
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"id": "msg_123", "type": "message", "role": "assistant", "model": "claude-3-5-sonnet-20241022", "content": [{"type": "text", "text": "Hello, world!"}], "usage": {"input_tokens": 10, "output_tokens": 10}}).encode('utf-8'))

    def do_GET(self):
        print("=== GET REQUEST ===")
        print(f"Path: {self.path}")
        print("Headers:")
        for k, v in self.headers.items():
            print(f"  {k}: {v}")
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b"{}")

httpd = HTTPServer(('localhost', 8089), SimpleHTTPRequestHandler)
print("Listening on port 8089...")
httpd.serve_forever()
