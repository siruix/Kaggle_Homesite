# This is a EC2 wappter that start my EC2 instance, run the program and stop the instance

# instance id
#INSTANCE_ID=i-a020bc79
INSTANCE_ID=i-730e91aa
VOLUMN_ID=vol-ae0e2b6f
# attach data volumn
aws ec2 attach-volume --volume-id $VOLUMN_ID --instance-id $INSTANCE_ID --device /dev/sdf