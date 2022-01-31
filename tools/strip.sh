#!/bin/bash

rm -rfv 3rd-party/ruy/third_party/googletest/
rm -rfv extras
(cd 3rd-party/ruy/third_party/cpuinfo/ && git pull https://github.com/browsermt/cpuinfo)
