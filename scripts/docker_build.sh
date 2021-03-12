#!/bin/bash

echo "Make sure you call the script from the root of this project "

docker build -f Dockerfile -t surprise:snapshot .