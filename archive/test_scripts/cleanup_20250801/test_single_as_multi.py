import modal

app = modal.App.from_name("multitalk-cuda")
generate_multi_person_video = modal.Function.from_name("generate_multi_person_video", app=app)

result = generate_multi_person_video.remote(
    prompt="A person speaking naturally with clear lip sync",
    image_key="multi1.png", 
    audio_keys=["1.wav"],  # Single audio as list
    sample_steps=20,
    output_prefix="single_as_multi"
)

print(result)
