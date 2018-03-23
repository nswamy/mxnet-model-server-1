# MXNet Model Server in Production

## Contents
* [Overview](#overview)
* [Running MMS In Container](#running-mms-in-container)
* [Launching MMS in ECS](#launching-mms-in-ecs)
* [Optimized Configuration Settings](#optimized-configuration-settings)

## Overview
MXNet Model Server *(MMS)* is a flexible and scalable serving tool to load and serve deep-learning models. The server creates
and exposes per-model endpoints and clients can use these endpoints to run their inference requests. If run manually,
this tool runs as a stand-alone mode. For production deployments, we recommend the container'ized MMS. This has multiple
advantages from efficient system utilization to ease of deployment.

In this document we cover the following:

    a. Running MXNet Model Server in a docker container.
    b. Deploying these containers in AWS ECS.
    c. Performance results for these MMS Docker instances.
    
## Running MMS In Container
Let's start with the basics of downloading and running MMS in a container setup. We currently host the latest MMS docker 
images encapsulating the latest changes from our github repository on docker-hub registry. There are two sets of 
docker images available on docker-hub registry, one for [CPU](https://hub.docker.com/r/awsdeeplearningteam/mms_cpu/) and 
one for [GPU](https://hub.docker.com/r/awsdeeplearningteam/mms_gpu/). These images are readily deployable on any host which
has a `docker` daemon running. 

### Getting the MMS Docker image
To get the image, follow the steps mentioned in the docker-hub registry. In this example we will follow setting up MMS
on a personal computer. For this we use the CPU image. To run a GPU version of the MMS, follow the documentation 
provided in the [Readme](../docker/README.md) section in the docker folder.

Make sure you have a `docker` daemon is up and running. Run 
```bash
docker pull awsdeeplearningteam/mms_cpu
```

This command gets the image from the docker-hub repository onto the localhost. To verify that the image was downloaded
you could run 
```bash
docker images
```
This should display all the images, which are ready to be run.
```text
REPOSITORY                    TAG                 IMAGE ID            CREATED             SIZE
awsdeeplearningteam/mms_cpu   latest              ce21d21baeab        8 days ago          1.71GB
ubuntu                        latest              f975c5035748        2 weeks ago         112MB
```

### Preparing the model files that need to be served
We need to create a local volume which would be hold the archived `model` files and the MMS configuration file (`mms.conf`)
which are required to run the MMS. Lets create a new folder called `mms_dir` and copy the [mms.conf](../docker/mms_app_cpu.conf) 
into the folder.
```bash
mkdir /home/user/mms_dir
```
```bash
cp ~/mxnet-model-server/docker/mms_app_cpu.conf /home/user/mms_dir/.
``` 
This configuration file contains all the configurations that are required to run the MXNet Model Server.
Modify the `--models` parameter and update it to point to models that need to be served. 

##### Example of having a local archived model file 
Lets say we want to serve `resnet-18` which is downloaded onto the localhost. We would place this in the `mms_dir` that
we created above. 
```bash
ls /home/user/mms_dir
```
```text
total 0
-rw-r--r--  1 user  user  0 Mar 20 01:32 mms.conf
-rw-r--r--  1 user  user  0 Mar 20 01:32 resnet-18.model
```
Next, we should update the `--models` parameter in the `mms.conf` file to reflect this `resnet-18` model.
```text
[MMS Arguments]
--models
resnet-18=/models/resnet-18.model
...
```
This configuration file reflects the path inside the container that we will run MMS in, which contains this model file.
This is updated as such because `mms_dir` is volume mounted onto the docker instance at `/models`. We will see this in 
the following section.

##### Example of having a model file downloaded from S3
MMS also supports pulling the model file from S3. The model configuration file `mms.conf` would look as follows in this
scenario.

```text
[MMS Arguments]
--models
resnet-18=https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model
...
```

### Running MMS
Running the MMS is done in two stages. One to run the docker image and then start the MMS.

#### Running the image
Once the image is downloaded, we can go ahead and run the MMS inside the docker instance. The following command will
run the docker image in an interactive but detached mode. Since the `nginx` in the `mms.conf` file is configured to 
listen on port 8080, we will expose this container port and map it to host port `80`.

```bash
docker run --name mms -p 80:8080 -itd -v /home/user/mms_dir:/models awsdeeplearningteam/mms_cpu
```

#### Starting MMS inside the container
To start the MMS inside the container instance, run the following command.
```bash
docker exec mms bash -c "mxnet-model-server.sh start --mms-conf /models/mms.conf"
```
This command would start the MXNet model server inside the container. Now we can run inferences against the localhost.


### Running inference
To run inference and test the output, you could follow the following example
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://localhost:80/squeezenet/predict -F "data=@kitten.jpg"
```

This would return a result similar to the following,

```text
{
  "prediction": [
    [
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.9408261179924011
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.055966004729270935
      },
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.0025502564385533333
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.00034320182749070227
      },
      {
        "class": "n02123394 Persian cat",
        "probability": 0.00026897044153884053
      }
    ]
  ]
}
``` 

### Update models and restart the MMS
In order to update the models that need to be served, you could update the `/home/user/mms.conf` folder. Update the 
`--models` parameter here to reflect all the models that need to be served. Once you are finished with this updation,
run the following command
Example: Lets add a new model `squeezenet` to be served,

```text
[MMS Arguments]
--models
resnet-18=https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model

```
```bash
docker exec mms bash -c "mxnet-model-server.sh restart --mms-conf /models/mms.conf" 
```

This would update the current docker instance of MMS with the updated model endpoints. 


## Launching MMS in ECS
Now that we have seen how to launch MMS in a container, we can now look at scaling this architecture for production.
In this document, we will look into a setup which looks as follows:

![alt text](https://s3.amazonaws.com/mms-doc-images/gen-architecture.jpg "MMS Architecture Diagram")

Here we created a ECS cluster with one VPC and two availability zones. Each of these zones have their own subnet's. 

We utilize the ECS features to store the container images and deploy these images on the cluster's instances. In this document

Pre-requisites for this following setup is that you already have a AWS account and you have created Approximate time required to create your own production (less than 2 hours).

The steps to create a MMS cluster is split into the following sections: 
* [Build and/or push Docker Image onto ECR repository](#build-and-publish-a-docker-image-onto-amazon-ecr-repository)
* [Create an ECS Cluster](#create-a-ecs-cluster)
* [Create an EFS](#create-an-efs-for-persistent-storage-across-cluster-instances)
* [Create ECS Cluster Task](#create-tasks-in-the-cluster)
* [Create an ELB](#create-a-elastic-load-balancer)
* [Starting MMS Service](#starting-a-service)
* [Verification of the MMS Service](#verification-of-mms-service-health)

### Build and publish a Docker image onto Amazon ECR Repository
After logging onto your AWS account, 
     
     1. Go to "Services" and select "Elastic Container Service". TODO: Update image (Go To Services->ECS)
     2. Click on "Repositories". 
     3. Click on "Create Repository". Here
         a. Fill in Respository name. Eg. "mms-cpu", go to "Next Step".
         b. Here you will find instructions to push the Docker image to Amazon ECR Repository.
            (NOTE: If you have pulled the image from the docker-hub, you wouldn't need to do the docker build. 
            Simply tag the image that you downloaded from docker hub with the repository's tag as shown in the commands.)
     TODO: Update image (ECR Repository)
     
Once the image is pushed to the ECR repository, make a note of the Repository URI: Eg: **9682771668.dkr.ecr.us-east-1.amazonaws.com/mms-cpu** 
     
### Create a ECS Cluster
After creating an image in the Amazon ECR repository, 
    
    1. Click on the `Clusters`
    2. Select `Create Cluster`. For this example, lets create a "EC2 Linux + Networking" template. TODO: Image (cluster/select_template.png) and click on "Next step"
        a. Give a cluster name. Eg "mms-cpu-cluster"
        b. Provisioning model is "On-Demand Instance"
        c. Select an EC2 instance type. Eg: C4.2xlarge.
        d. Select the number of instances. Eg: 4
        e. Select  amount of EBS Storage. Eg: 80 GiB
        f. Select "Key-pair". Its recommended that you select a key-pair so that you would be able to log into the EC2 instances in the future.
        g. Networking: For this example I will be creating a new VPC with two subnets. If you want to use an existing VPC, you could reuse that.
            i.    VPC: "Create a new VPC"
            ii.   CIDR Block : 10.0.0.0/16
            iii.  Subnet 1   : 10.0.0.0/24
            iv.   Subnet 2   : 10.0.1.0/24
            v.    Security Group: Create a new Security group.
            vi.   Security group Inbound Rules: Default.
        h. Container instance IAM Role: ecsInstanceRole.
        i. Then hit "create"
        
This creates the cluster. Once the cluster is created, go to "View Cluster" and select the cluster that you created.
Go to the "EC2 Instances" tab. We will need to make a note of the VPC that we created before we proceed to the next 
section. In our example, the newly created VPC was `"vpc-1a47cf61"`

### Create an EFS for persistent storage across Cluster instances
Similar to the stand-alone model, we will be creating a persistent storage across the EC2 instances to hold the
`mms.conf` file and the `model` files. These can be used by all EC2 instances across the cluster.

    1. Go to Services on the account tab and search for EFS. Select that service. TODO: Image 
    2. Click on "Create file system"
    3. Select the VPC that this filesystem would be accessed from. NOTE: Use the VPC that your cluster is working in here. 
       TODO: Image In our example we will be selecting "vpc-1a47cf61" as found in the above section. TODO" EFS_step1.png
    4. Goto Next step.
    5. In this section you could select Tags, Performance Modes and Enable Encryption. TODO: Image EFS_step2.png
    6. Create the File System. TODO: Image EFS_created.png
    
**NOTE: Before proceeding to the next step, click on "Amazon EC2 mount instructions" in the created EFS section 
and follow the instructions provided to mount this EFS onto the EC2 instances.**

Issues that can be faced:
1. If you face issues logging into the EC2 instances:
     You might have to modify the "security group" associated with the EC2 instance and add an 
     "Inbound" rule to accept all SSH Traffic. TODO: Image enable_ssh_ec2.png
2. If you face issues with mounting the file system onto `/efs`:
   a. Add a inbound rule in the EC2-Instance-Security-Group to accept all traffic from the EFS-Security group.
   b. Similarly, add an inbound rule to EFS-Security-group to accept all traffic from the EC2-Instance-Security-Group.

Refer [EFS Document](https://docs.aws.amazon.com/efs/latest/ug/getting-started.html) for further information.

At the end of this procedure, each of your EC2 instances would have this EFS mounted onto their filesystem at "/efs" 
directory. 

For the sake of legibility, we will create a folder called "models" inside the "/efs" directory on the EC2 instance.
So we would have "/efs/models" on each of the EC2 instances. For this example, we will copy [mms_app_cpu.conf](../docker/mms_app_cpu.conf)
into this folder which we will use later when we create the service.

### Create Tasks in the cluster
In order to run our container in the cluster, we would have to create a task. This task can be spun-up as a service 
in our cluster. In this section we will see how we can create a task to start "mxnet-model-server" in our cluster.

    1. Go to Services and select "Elastic Container Service".
    2. Click on "Task Definitions". TODO: Image task_landing.png
    3. Click on "Create new Task Definition".
    4. Select the launch type compatibility. For this example, we have selected "EC2".
    5. Click on "Next Step". 
    6. In this page we will have to fill in the details of the task.
       a. Task Definition Name* . Eg: mms-cpu-demo-task
       b. Requires Compatibilities* EC2
       c. Task Role : ecsTaskExecutionRole
       d. Network Mode: Bridge
       e. Task Execution Role: ecsTaskExecutionRole 
       f. Task Memory: For this example we will use 10GB (or 10000)
       g. Task CPU: For this example we will use 2 vCPU
       h. Constraints : Default
       i. Volumes: We will add volumes. 
           i.   Name: efs
           ii.  Source Path: /efs/models
       j. Then we will click on "Add container".
           i.   Container Name: Eg: mms-cpu-container
           ii.  Image* : The Repository URI from above "9682771668.dkr.ecr.us-east-1.amazonaws.com/mms-cpu".
           iii. Memory: For this example lets take 2000MiB (or 2GiB) and a soft limit.
           iv.  Port mapping: Map the container ports to the host port. Since we will use default mms_app_cpu.conf here, 
                We will map port "80" on host to "8080" on container port. 
           v. In Advanced Container Configuration we will only fill in "Environment", "Storage and Logging" and Security.
               a. TODO: task_environment.png
               b. TODO: task_storage_logging.png
               c. TODO: task_security.png
               d. Click on "Add".
           vi. Click on "Create".

Now we have a task that can be started on our cluster.


### Create a Elastic Load Balancer
Production traffic would need to be distributed across instances. For this we are going to use ELB or Elastic Load Balancers.
    
    1. Go to Services and select EC2. TODO: service_ec2.png
    2. Select Load Balancers. TODO: load-balancing.png
    3. Click on "Create Load Balancer".
        a. Here we will be creating an Application Load Balancer (ALB). Select Application Load Balancer.
        b. We will create ALB named "mms-demo-elb" which is an "internet-facing" load-balancer which works in the VPC
           that we had created for our cluster. In our example "vpc-1a47cf61". Go to the next section.
           TODO: configure-load-balancing.png
    4. Since we don't have a HTTPS listener, we will be ignoring the "Configure Security Settings" sections and will skip
       forward to "Configure Security Groups" section.
    5. Configure Security Groups. Here we will create a new security group. 
       TODO: configure-security-group.png.
    6. Next we will create Routing for the ALB and go to "Register Targets" section. TODO: configure-routing.png
    7. Here since we have only one target and we are not doing path based routing, we will have all the instance in the
       same target. Next we Review TODO: register-targets.png
    8. Once you are reviewed the changes, go ahead and create this ALB.
    
After the ALB is created, we would need to add an inbound rule to the "EC2-instances-SG" to allow all traffic from the "ALB-SG".

In our example, the Application Load Balancer's Security group ID is "sg-f6abfd80" and EC2 instances Security Group ID is "sg-7b194f0d".

    1. Click on the EC2 instance security-group.
    2. Add inbound rule to accept all traffic from "sg-f6abfd80", which was our Application Load Balancers Security Group ID.
        TODO: add-elb-sg-to-ec2-sg.png

Now that we have an application load balancer, we can go ahead and create our service.
    
### Starting MMS service
Once the task is created, we can now create a Service that can be launched onto the ECS cluster. 
TODO: Image "create_service.png"

    1. Fill in the Configure service portion. Lets use create 4 tasks in this example.
       TODO: Image "service_step1.png"
    2. Next, Configure the Network for the service.
       TODO: Image "configure-network.png"
    3. Next, Set Auto-scaling for the service. 
       TODO: auto-scaling.png.
    4. Next, Review the changes and create a service.
    
This would have created a service which would be up and running. 

### Verification of MMS Service Health
 
Verify that the service is up and running by going to the browser and hitting the Application Load Balancers endpoint. If the Application
Load Balancer's DNS Name is mms-demo-elb-353755180.us-east-1.elb.amazonaws.com,

```text
http://mms-demo-elb-353755180.us-east-1.elb.amazonaws.com/ping
```

You should get a response as follows
```text
{
  "health": "healthy!"
}
```

You could run inferences against this endpoint now. 


### Updating models and re-deployments

## Performance metrics