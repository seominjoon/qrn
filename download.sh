#!/usr/bin/env bash

SOURCE_DIR="$HOME/data"
NAME="tasks_1-20_v1-2"
BABI_DIR="$SOURCE_DIR/babi"
mkdir $SOURCE_DIR
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz -O $SOURCE_DIR/babi.tar.gz
tar -zxvf $SOURCE_DIR/babi.tar.gz -C $SOURCE_DIR
mv $SOURCE_DIR/$NAME $BABI_DIR