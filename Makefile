develop:
	gradio app.py

production:
	docker run --rm -it -p 7860:7860 code-interpreter-lite

full:
	docker compose -f docker-compose.yml up

full-detached:
	docker compose -f docker-compose.yml up -d
	docker compose logs -f

build-production-image:
	docker build -t code-interpreter-lite:latest .

build-development-environment:
	pip install -r requirements.txt