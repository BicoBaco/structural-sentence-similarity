#!/bin/sh

sentence=$1
name=$2

cd candc
echo "$sentence" | bin/candc --models models/boxer --candc-printer boxer > "working/$name.ccg"

bin/boxer --input "working/$name.ccg" --output "working/$name.drg" --semantics drg 