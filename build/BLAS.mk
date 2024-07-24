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

CXX_FLAGS := 
C_FLAGS += -Wall 
C_FLAGS += -Werror=return-type
C_FLAGS += -fvisibility=hidden
C_FLAGS += -fPIC
CXX_FLAGS += -pthread
ifeq (true, $(APP_DEBUG))
	CXX_FLAGS += -g -O0 -UNDEBUG
else
	CXX_FLAGS += -O2 -DNDEBUG
endif
CXX_FLAGS += -I$(THIRD_PARTY_DIR)/OpenBLAS/include
CXX_FLAGS += -I$(THIRD_PARTY_DIR)/OpenBLAS/include/openblas/linux

LD_FLAGS := 
LD_FLAGS += -pthread
LD_FLAGS += -Wl,--no-undefined
LD_FLAGS += -Wl,-rpath,\$$ORIGIN
LD_FLAGS += -z now
LD_FLAGS += -z relro
ifneq (true, $(APP_DEBUG))
	LD_FLAGS += -s
endif

all : \
	$(BIN_DIR)/BLAS-Linear-Regression-Normal-Equation \
	$(BIN_DIR)/BLAS-Linear-Regression-Gradient-Descent \
	$(BIN_DIR)/BLAS-Logistic-Regression

# Link
$(BIN_DIR)/BLAS-Linear-Regression-Normal-Equation: $(OBJ_DIR)/BLAS-Linear-Regression-normal-equation-main.o $(BIN_DIR)/libopenblasp.so $(BIN_DIR)/libgfortran.so $(BIN_DIR)/libquadmath.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/BLAS-Linear-Regression-normal-equation-main.o -L$(BIN_DIR) -lopenblasp -lgfortran -lquadmath -o $(BIN_DIR)/BLAS-Linear-Regression-Normal-Equation

$(BIN_DIR)/BLAS-Linear-Regression-Gradient-Descent: $(OBJ_DIR)/BLAS-Linear-Regression-gradient-descent-main.o $(BIN_DIR)/libopenblasp.so $(BIN_DIR)/libgfortran.so $(BIN_DIR)/libquadmath.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/BLAS-Linear-Regression-gradient-descent-main.o -L$(BIN_DIR) -lopenblasp -lgfortran -lquadmath -o $(BIN_DIR)/BLAS-Linear-Regression-Gradient-Descent

$(BIN_DIR)/BLAS-Logistic-Regression: $(OBJ_DIR)/BLAS-Logistic-Regression-main.o $(BIN_DIR)/libopenblasp.so $(BIN_DIR)/libgfortran.so $(BIN_DIR)/libquadmath.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/BLAS-Logistic-Regression-main.o -L$(BIN_DIR) -lopenblasp -lgfortran -lquadmath -o $(BIN_DIR)/BLAS-Logistic-Regression

# Compile
$(OBJ_DIR)/BLAS-Linear-Regression-normal-equation-main.o: $(SOURCE_DIR)/BLAS/Linear-Regression/normal-equation-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(CXX_FLAGS) $(SOURCE_DIR)/BLAS/Linear-Regression/normal-equation-main.cpp -MD -MF $(OBJ_DIR)/BLAS-Linear-Regression-Normal-Equation-main.d -o $(OBJ_DIR)/BLAS-Linear-Regression-normal-equation-main.o

$(OBJ_DIR)/BLAS-Linear-Regression-gradient-descent-main.o: $(SOURCE_DIR)/BLAS/Linear-Regression/gradient-descent-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(CXX_FLAGS) $(SOURCE_DIR)/BLAS/Linear-Regression/gradient-descent-main.cpp -MD -MF $(OBJ_DIR)/BLAS-Linear-Regression-Gradient-Descent-main.d -o $(OBJ_DIR)/BLAS-Linear-Regression-gradient-descent-main.o

$(OBJ_DIR)/BLAS-Logistic-Regression-main.o: $(SOURCE_DIR)/BLAS/Logistic-Regression/main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(CXX_FLAGS) $(SOURCE_DIR)/BLAS/Logistic-Regression/main.cpp -MD -MF $(OBJ_DIR)/BLAS-Logistic-Regression-main.d -o $(OBJ_DIR)/BLAS-Logistic-Regression-main.o

-include \
	$(OBJ_DIR)/BLAS-Linear-Regression-Normal-Equation-main.d \
	$(OBJ_DIR)/BLAS-Linear-Regression-Gradient-Descent-main.d \
	$(OBJ_DIR)/BLAS-Logistic-Regression-main.d

# Copy
$(BIN_DIR)/libopenblasp.so: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libopenblasp.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libopenblasp.so $(BIN_DIR)/libopenblasp.so

$(BIN_DIR)/libgfortran.so: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libgfortran.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libgfortran.so $(BIN_DIR)/libgfortran.so

$(BIN_DIR)/libquadmath.so: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libquadmath.so
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libquadmath.so $(BIN_DIR)/libquadmath.so

clean:
	$(HIDE) rm -f $(BIN_DIR)/BLAS-Linear-Regression-Normal-Equation
	$(HIDE) rm -f $(BIN_DIR)/BLAS-Linear-Regression-Gradient-Descent
	$(HIDE) rm -f $(BIN_DIR)/BLAS-Logistic-Regression
	$(HIDE) rm -f $(OBJ_DIR)/BLAS-Linear-Regression-normal-equation-main.o
	$(HIDE) rm -f $(OBJ_DIR)/BLAS-Linear-Regression-gradient-descent-main.o
	$(HIDE) rm -f $(OBJ_DIR)/BLAS-Logistic-Regression-main.o
	$(HIDE) rm -f $(OBJ_DIR)/BLAS-Linear-Regression-Normal-Equation-main.d
	$(HIDE) rm -f $(OBJ_DIR)/BLAS-Linear-Regression-Gradient-Descent-main.d
	$(HIDE) rm -f $(OBJ_DIR)/BLAS-Logistic-Regression-main.d
	$(HIDE) rm -f $(BIN_DIR)/libopenblasp.so
	$(HIDE) rm -f $(BIN_DIR)/libgfortran.so
	$(HIDE) rm -f $(BIN_DIR)/libquadmath.so

.PHONY : \
	all \
	clean
