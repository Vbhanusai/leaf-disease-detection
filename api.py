from gradio_client import Client, handle_file

client = Client("bhanusAI/plantifysol")
result = client.predict(
		img=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		plant_type="Apple",
		api_name="/predict"
)
print(result)
