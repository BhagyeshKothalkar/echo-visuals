from locust import HttpUser, between, task


class Flux2User(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def generate_image(self):
        payload = {
            "model": "black-forest-labs/FLUX.2-klein-4B",
            "prompt": "a photorealistic red fox standing in fresh snow",
            "size": "1024x1024",
            "num_inference_steps": 4,
            "response_format": "b64_json",
            "seed": 42,
        }

        with self.client.post(
            "/v1/images/generations",
            json=payload,
            name="FLUX.2 Klein 4B",
            timeout=120,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:500]}")
