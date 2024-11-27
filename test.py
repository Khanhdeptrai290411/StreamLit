import base64

with open("static/logo.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
print(f"data:image/png;base64,{encoded_string}")
