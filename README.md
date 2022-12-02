# GCN_Docking_Scalability

## Build Docker
1. Create the Docker image `docker build -t your_image_name .`
2. Run your image `docker run -it -d --shm-size 128g your_image_name`
3. Access to your container `docker exec -it <container id> bash`