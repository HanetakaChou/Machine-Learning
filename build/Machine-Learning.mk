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
CODE_DIR := $(LOCAL_PATH)/../code
THIRD_PARTY_DIR := $(LOCAL_PATH)/../third-party

CXX_FLAGS := 
CXX_FLAGS += -Wall -Werror=return-type
CXX_FLAGS += -fvisibility=hidden
CXX_FLAGS += -fPIE -fPIC
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
LD_FLAGS += -Wl,--enable-new-dtags -Wl,-rpath,'$$ORIGIN'
ifneq (true, $(APP_DEBUG))
	LD_FLAGS += -s
endif

all : \
	$(BIN_DIR)/Machine-Learning-Linear-Regression-Normal-Equation \
	$(BIN_DIR)/Machine-Learning-Linear-Regression-Gradient-Descent \
	$(BIN_DIR)/Machine-Learning-Logistic-Regression

# Link
$(BIN_DIR)/Machine-Learning-Linear-Regression-Normal-Equation: $(OBJ_DIR)/Machine-Learning-Linear-Regression-normal-equation-main.o $(BIN_DIR)/libopenblasp.so.0 $(BIN_DIR)/libgfortran.so.5 $(BIN_DIR)/libquadmath.so.0
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/Machine-Learning-Linear-Regression-normal-equation-main.o $(BIN_DIR)/libopenblasp.so.0 $(BIN_DIR)/libgfortran.so.5 $(BIN_DIR)/libquadmath.so.0 -o $(BIN_DIR)/Machine-Learning-Linear-Regression-Normal-Equation

$(BIN_DIR)/Machine-Learning-Linear-Regression-Gradient-Descent: $(OBJ_DIR)/Machine-Learning-Linear-Regression-gradient-descent-main.o $(BIN_DIR)/libopenblasp.so.0 $(BIN_DIR)/libgfortran.so.5 $(BIN_DIR)/libquadmath.so.0
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/Machine-Learning-Linear-Regression-gradient-descent-main.o $(BIN_DIR)/libopenblasp.so.0 $(BIN_DIR)/libgfortran.so.5 $(BIN_DIR)/libquadmath.so.0 -o $(BIN_DIR)/Machine-Learning-Linear-Regression-Gradient-Descent

$(BIN_DIR)/Machine-Learning-Logistic-Regression: $(OBJ_DIR)/Machine-Learning-Logistic-Regression-main.o $(BIN_DIR)/libopenblasp.so.0 $(BIN_DIR)/libgfortran.so.5 $(BIN_DIR)/libquadmath.so.0
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) clang++ -pie $(LD_FLAGS) $(OBJ_DIR)/Machine-Learning-Logistic-Regression-main.o $(BIN_DIR)/libopenblasp.so.0 $(BIN_DIR)/libgfortran.so.5 $(BIN_DIR)/libquadmath.so.0 -o $(BIN_DIR)/Machine-Learning-Logistic-Regression

# Compile
$(OBJ_DIR)/Machine-Learning-Linear-Regression-normal-equation-main.o: $(CODE_DIR)/Machine-Learning/Linear-Regression/normal-equation-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(CXX_FLAGS) $(CODE_DIR)/Machine-Learning/Linear-Regression/normal-equation-main.cpp -MD -MF $(OBJ_DIR)/Machine-Learning-Linear-Regression-Normal-Equation-main.d -o $(OBJ_DIR)/Machine-Learning-Linear-Regression-normal-equation-main.o

$(OBJ_DIR)/Machine-Learning-Linear-Regression-gradient-descent-main.o: $(CODE_DIR)/Machine-Learning/Linear-Regression/gradient-descent-main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(CXX_FLAGS) $(CODE_DIR)/Machine-Learning/Linear-Regression/gradient-descent-main.cpp -MD -MF $(OBJ_DIR)/Machine-Learning-Linear-Regression-Gradient-Descent-main.d -o $(OBJ_DIR)/Machine-Learning-Linear-Regression-gradient-descent-main.o

$(OBJ_DIR)/Machine-Learning-Logistic-Regression-main.o: $(CODE_DIR)/Machine-Learning/Logistic-Regression/main.cpp
	$(HIDE) mkdir -p $(OBJ_DIR)
	$(HIDE) clang++ -c $(CXX_FLAGS) $(CODE_DIR)/Machine-Learning/Logistic-Regression/main.cpp -MD -MF $(OBJ_DIR)/Machine-Learning-Logistic-Regression-main.d -o $(OBJ_DIR)/Machine-Learning-Logistic-Regression-main.o

-include \
	$(OBJ_DIR)/Machine-Learning-Linear-Regression-Normal-Equation-main.d \
	$(OBJ_DIR)/Machine-Learning-Linear-Regression-Gradient-Descent-main.d \
	$(OBJ_DIR)/Machine-Learning-Logistic-Regression-main.d

# Copy
$(BIN_DIR)/libopenblasp.so.0: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libopenblasp.so.0
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libopenblasp.so.0 $(BIN_DIR)/libopenblasp.so.0

$(BIN_DIR)/libgfortran.so.5: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libgfortran.so.5
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libgfortran.so.5 $(BIN_DIR)/libgfortran.so.5

$(BIN_DIR)/libquadmath.so.0: $(THIRD_PARTY_DIR)/OpenBLAS/lib/libquadmath.so.0
	$(HIDE) mkdir -p $(BIN_DIR)
	$(HIDE) cp -f $(THIRD_PARTY_DIR)/OpenBLAS/lib/libquadmath.so.0 $(BIN_DIR)/libquadmath.so.0

clean:
	$(HIDE) rm -f $(BIN_DIR)/Machine-Learning-Linear-Regression-Normal-Equation
	$(HIDE) rm -f $(BIN_DIR)/Machine-Learning-Linear-Regression-Gradient-Descent
	$(HIDE) rm -f $(BIN_DIR)/Machine-Learning-Logistic-Regression
	$(HIDE) rm -f $(OBJ_DIR)/Machine-Learning-Linear-Regression-normal-equation-main.o
	$(HIDE) rm -f $(OBJ_DIR)/Machine-Learning-Linear-Regression-gradient-descent-main.o
	$(HIDE) rm -f $(OBJ_DIR)/Machine-Learning-Logistic-Regression-main.o
	$(HIDE) rm -f $(OBJ_DIR)/Machine-Learning-Linear-Regression-Normal-Equation-main.d
	$(HIDE) rm -f $(OBJ_DIR)/Machine-Learning-Linear-Regression-Gradient-Descent-main.d
	$(HIDE) rm -f $(OBJ_DIR)/Machine-Learning-Logistic-Regression-main.d
	$(HIDE) rm -f $(BIN_DIR)/libopenblasp.so.0
	$(HIDE) rm -f $(BIN_DIR)/libgfortran.so.5
	$(HIDE) rm -f $(BIN_DIR)/libquadmath.so.0

.PHONY : \
	all \
	clean