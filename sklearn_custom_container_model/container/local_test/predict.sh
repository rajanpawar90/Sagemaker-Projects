#!/bin/bash

payload=$1
content=${2:-text/csv}

curl --data binary @${payload} -H "content-Type: ${content}" -v http://localhost:8080/invocations