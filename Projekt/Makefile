BUILD    := build           # alla .o + exe hit
TARGET   := $(BUILD)/ssa_seq

SRC_CPP  := $(wildcard src/*.cpp)
SRC_C    := $(wildcard src/*.c)

OBJ      := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRC_CPP)) \
            $(patsubst src/%.c,  $(BUILD)/%.o,$(SRC_C))

# skapa build/ om den saknas
$(shell mkdir -p $(BUILD))

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@

$(BUILD)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.c
	$(CC)  $(CFLAGS)  -c $< -o $@

clean:
	rm -rf $(BUILD)