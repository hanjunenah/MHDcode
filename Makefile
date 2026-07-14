CXX ?= c++
PKG_CONFIG ?= pkg-config
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS += $(shell $(PKG_CONFIG) --cflags hdf5)
LDLIBS += $(shell $(PKG_CONFIG) --libs hdf5) -pthread

TARGET := mhd2d
SOURCES := src/main.cpp

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(LDLIBS) -o $@

test: $(TARGET)
	python3 run.py input/harris.py --t-end 0.1 --threads 2 --output output/smoke.h5 --force
	MPLBACKEND=Agg python3 plot_results.py output/smoke.h5 --frame -1 --save output/smoke.png
	python3 run.py input/alfven_antenna.py --t-end 0.1 --threads 2 \
		--output output/antenna_smoke.h5 --force
	MPLBACKEND=Agg python3 plot_antenna_results.py output/antenna_smoke.h5 \
		--save output/antenna_smoke.png
	python3 run.py input/orszag_tang.py --t-end 0.01 --threads 2 \
		--output output/orszag_tang_smoke.h5 --force

clean:
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM
	rm -rf output __pycache__ input/__pycache__
