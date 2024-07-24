#
# Copyright (C) YuqiaoZhang(HanetakaChou)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

HIDE := @

LOCAL_PATH := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
ifeq (true, $(APP_DEBUG))
	BIN_DIR := $(LOCAL_PATH)/bin/debug
	OBJ_DIR := $(LOCAL_PATH)/obj/debug
else
	BIN_DIR := $(LOCAL_PATH)/bin/release
	OBJ_DIR := $(LOCAL_PATH)/obj/release
endif
SOURCE_DIR := $(LOCAL_PATH)/../source
THIRD_PARTY_DIR := $(LOCAL_PATH)/../thirdparty

C_FLAGS := 
C_FLAGS += -Wall 
C_FLAGS += -Werror=return-type
C_FLAGS += -fvisibility=hidden
C_FLAGS += -fPIC
C_FLAGS += -pthread
ifeq (true, $(APP_DEBUG))
	C_FLAGS += -g -O0 -UNDEBUG
else
	C_FLAGS += -O2 -DNDEBUG
endif
C_FLAGS += -I$(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/include
C_FLAGS += -I$(THIRD_PARTY_DIR)/TensorFlow-Lite/include
C_FLAGS += -I$(THIRD_PARTY_DIR)/OpenBLAS/include
C_FLAGS += -I$(THIRD_PARTY_DIR)/OpenBLAS/include/openblas/linux
C_FLAGS += -I$(THIRD_PARTY_DIR)/Python3/include
C_FLAGS += -I$(THIRD_PARTY_DIR)/Python3/include/linux

LD_FLAGS := 
LD_FLAGS += -pthread
LD_FLAGS += -Wl,--no-undefined
LD_FLAGS += -Wl,--enable-new-dtags 
LD_FLAGS += -Wl,-rpath,\$$ORIGIN
LD_FLAGS += -z now
LD_FLAGS += -z relro
ifneq (true, $(APP_DEBUG))
	LD_FLAGS += -s
endif

all : \
	$(BIN_DIR)/TensorFlow-Neural-Network \
	$(BIN_DIR)/TensorFlow-Multinomial-Logistic-Regression \
	$(BIN_DIR)/TensorFlow-Regression-Based-Collaborative-Filtering \
	$(BIN_DIR)/TensorFlow-Classification-Based-Collaborative-Filtering \
	$(BIN_DIR)/TensorFlow-Reinforcement-Learning

# Link
$(BIN_DIR)/TensorFlow-Neural-Network: $(OBJ_DIR)/TensorFlow-Neural-Network-inference-main.o $(BIN_DIR)/libOpenCL.so $(BIN_DIR)/libtensorflowlite_c.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/TensorFlow-Neural-Network-inference-main.o -L$(BIN_DIR) -lOpenCL -ltensorflowlite_c -o $(BIN_DIR)/TensorFlow-Neural-Network

$(BIN_DIR)/TensorFlow-Multinomial-Logistic-Regression: $(OBJ_DIR)/TensorFlow-Multinomial-Logistic-Regression-inference-main.o $(BIN_DIR)/libOpenCL.so $(BIN_DIR)/libtensorflowlite_c.so $(BIN_DIR)/libopenblasp.so $(BIN_DIR)/libgfortran.so $(BIN_DIR)/libquadmath.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/TensorFlow-Multinomial-Logistic-Regression-inference-main.o -L$(BIN_DIR) -lOpenCL -ltensorflowlite_c -lopenblasp -lgfortran -lquadmath -o $(BIN_DIR)/TensorFlow-Multinomial-Logistic-Regression

$(BIN_DIR)/TensorFlow-Regression-Based-Collaborative-Filtering: $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-regression-inference-main.o $(BIN_DIR)/libOpenCL.so $(BIN_DIR)/libtensorflowlite_c.so $(BIN_DIR)/libopenblasp.so $(BIN_DIR)/libgfortran.so $(BIN_DIR)/libquadmath.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-regression-inference-main.o -L$(BIN_DIR) -lOpenCL -ltensorflowlite_c -lopenblasp -lgfortran -lquadmath -o $(BIN_DIR)/TensorFlow-Regression-Based-Collaborative-Filtering

$(BIN_DIR)/TensorFlow-Classification-Based-Collaborative-Filtering: $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-classification-inference-main.o $(BIN_DIR)/libOpenCL.so $(BIN_DIR)/libtensorflowlite_c.so $(BIN_DIR)/libopenblasp.so $(BIN_DIR)/libgfortran.so $(BIN_DIR)/libquadmath.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-classification-inference-main.o -L$(BIN_DIR) -lOpenCL -ltensorflowlite_c -lopenblasp -lgfortran -lquadmath -o $(BIN_DIR)/TensorFlow-Classification-Based-Collaborative-Filtering

$(BIN_DIR)/TensorFlow-Reinforcement-Learning : $(OBJ_DIR)/TensorFlow-Reinforcement-Learning-inference-main.o $(BIN_DIR)/libOpenCL.so $(BIN_DIR)/libtensorflowlite_c.so $(BIN_DIR)/libopenblasp.so $(BIN_DIR)/libgfortran.so $(BIN_DIR)/libquadmath.so $(BIN_DIR)/libpython39.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/TensorFlow-Reinforcement-Learning-inference-main.o -L$(BIN_DIR) -lOpenCL -ltensorflowlite_c -lopenblasp -lgfortran -lquadmath -lpython39 -o $(BIN_DIR)/TensorFlow-Reinforcement-Learning

$(BIN_DIR)/libOpenCL.so: $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux_envvars.o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux.o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch_generated.o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch.o $(OBJ_DIR)/OpenCL-ICD-Loader-icd.o
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -shared $(LD_FLAGS) -Wl,-soname,libOpenCL.so -Wl,--version-script=$(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/linux/icd_exports.map $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux_envvars.o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux.o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch_generated.o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch.o $(OBJ_DIR)/OpenCL-ICD-Loader-icd.o -o $(BIN_DIR)/libOpenCL.so

# Compile
$(OBJ_DIR)/TensorFlow-Neural-Network-inference-main.o: $(SOURCE_DIR)/TensorFlow/Neural-Network/inference-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(SOURCE_DIR)/TensorFlow/Neural-Network/inference-main.cpp -MD -MF $(OBJ_DIR)/TensorFlow-Neural-Network-inference-main.d -o $(OBJ_DIR)/TensorFlow-Neural-Network-inference-main.o

$(OBJ_DIR)/TensorFlow-Multinomial-Logistic-Regression-inference-main.o: $(SOURCE_DIR)/TensorFlow/Multinomial-Logistic-Regression/inference-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(SOURCE_DIR)/TensorFlow/Multinomial-Logistic-Regression/inference-main.cpp -MD -MF $(OBJ_DIR)/TensorFlow-Multinomial-Logistic-Regression-inference-main.d -o $(OBJ_DIR)/TensorFlow-Multinomial-Logistic-Regression-inference-main.o

$(OBJ_DIR)/TensorFlow-Collaborative-Filtering-regression-inference-main.o: $(SOURCE_DIR)/TensorFlow/Collaborative-Filtering/regression-inference-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(SOURCE_DIR)/TensorFlow/Collaborative-Filtering/regression-inference-main.cpp -MD -MF $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-regression-inference-main.d -o $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-regression-inference-main.o

$(OBJ_DIR)/TensorFlow-Collaborative-Filtering-classification-inference-main.o: $(SOURCE_DIR)/TensorFlow/Collaborative-Filtering/classification-inference-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(SOURCE_DIR)/TensorFlow/Collaborative-Filtering/classification-inference-main.cpp -MD -MF $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-classification-inference-main.d -o $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-classification-inference-main.o

$(OBJ_DIR)/TensorFlow-Reinforcement-Learning-inference-main.o: $(SOURCE_DIR)/TensorFlow/Reinforcement-Learning/inference-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(SOURCE_DIR)/TensorFlow/Reinforcement-Learning/inference-main.cpp -MD -MF $(OBJ_DIR)/TensorFlow-Reinforcement-Learning-inference-main.d -o $(OBJ_DIR)/TensorFlow-Reinforcement-Learning-inference-main.o

$(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux_envvars.o: $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/linux/icd_linux_envvars.c
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/linux/icd_linux_envvars.c -MD -MF $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux_envvars.d -o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux_envvars.o

$(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux.o: $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/linux/icd_linux.c
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/linux/icd_linux.c -MD -MF $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux.d -o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux.o

$(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch_generated.o: $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/icd_dispatch_generated.c
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/icd_dispatch_generated.c -MD -MF $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch_generated.d -o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch_generated.o

$(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch.o: $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/icd_dispatch.c
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/icd_dispatch.c -MD -MF $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch.d -o $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch.o

$(OBJ_DIR)/OpenCL-ICD-Loader-icd.o: $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/icd.c
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(C_FLAGS) $(THIRD_PARTY_DIR)/OpenCL-ICD-Loader/src/icd.c -MD -MF $(OBJ_DIR)/OpenCL-ICD-Loader-icd.d -o $(OBJ_DIR)/OpenCL-ICD-Loader-icd.o

-include \
	$(OBJ_DIR)/TensorFlow-Neural-Network-inference-main.d \
	$(OBJ_DIR)/TensorFlow-Multinomial-Logistic-Regression-inference-main.d \
	$(OBJ_DIR)/TensorFlow-Collaborative-Filtering-regression-inference-main.d \
	$(OBJ_DIR)/TensorFlow-Collaborative-Filtering-classification-inference-main.d \
	$(OBJ_DIR)/TensorFlow-Reinforcement-Learning-inference-main.d \
	$(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux_envvars.d \
	$(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux.d \
	$(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch_generated.d \
	$(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch.d \
	$(OBJ_DIR)/OpenCL-ICD-Loader-icd.d 

# Copy
$(BIN_DIR)/libtensorflowlite_c.so: $(THIRD_PARTY_DIR)/TensorFlow-Lite/lib/libtensorflowlite_c.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/TensorFlow-Lite/lib/libtensorflowlite_c.so $(BIN_DIR)/libtensorflowlite_c.so

$(BIN_DIR)/libopenblasp.so: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libopenblasp.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libopenblasp.so $(BIN_DIR)/libopenblasp.so

$(BIN_DIR)/libgfortran.so: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libgfortran.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libgfortran.so $(BIN_DIR)/libgfortran.so

$(BIN_DIR)/libquadmath.so: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libquadmath.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libquadmath.so $(BIN_DIR)/libquadmath.so

$(BIN_DIR)/libpython39.so: $(THIRD_PARTY_DIR)/Python3/lib/libpython39.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/Python3/lib/libpython39.so $(BIN_DIR)/libpython39.so	

clean:
	$(HIDE) rm -f $(BIN_DIR)/TensorFlow-Neural-Network
	$(HIDE) rm -f $(BIN_DIR)/TensorFlow-Multinomial-Logistic-Regression
	$(HIDE) rm -f $(BIN_DIR)/TensorFlow-Regression-Based-Collaborative-Filtering
	$(HIDE) rm -f $(BIN_DIR)/TensorFlow-Classification-Based-Collaborative-Filtering
	$(HIDE) rm -f $(BIN_DIR)/libOpenCL.so
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Neural-Network-inference-main.o
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Multinomial-Logistic-Regression-inference-main.o
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-regression-inference-main.o
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-classification-inference-main.o
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Reinforcement-Learning-inference-main.o
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux_envvars.o
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux.o
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch_generated.o
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch.o
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd.o
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Neural-Network-inference-main.d
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Multinomial-Logistic-Regression-inference-main.d
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-regression-inference-main.d
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Collaborative-Filtering-classification-inference-main.d
	$(HIDE) rm -f $(OBJ_DIR)/TensorFlow-Reinforcement-Learning-inference-main.d
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux_envvars.d
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd_linux.d
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch_generated.d
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd_dispatch.d
	$(HIDE) rm -f $(OBJ_DIR)/OpenCL-ICD-Loader-icd.d
	$(HIDE) rm -f $(BIN_DIR)/libtensorflowlite_c.so
	$(HIDE) rm -f $(BIN_DIR)/libopenblasp.so
	$(HIDE) rm -f $(BIN_DIR)/libgfortran.so
	$(HIDE) rm -f $(BIN_DIR)/libquadmath.so
	$(HIDE) rm -f $(BIN_DIR)/libpython39.so

.PHONY : \
	all \
	clean
