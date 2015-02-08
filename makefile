CXX      = clang++
LDFLAGS  = -lopencv_features2d -lopencv_core  -lopencv_highgui
CXXFLAGS = -std=c++11 -stdlib=libc++ -Wall

OBJDIR   = Build
SRCDIR   = Sources
SOURCES  = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS  = $(patsubst $(SRCDIR)/%, $(OBJDIR)/%, $(SOURCES:.cpp=.cpp.o))
TARGET   = Build/main
PARAMS	 = 

.PHONY: all clean	

all: $(OBJDIR) $(TARGET)
	@echo "${GREEN}Build complete. Running $(TARGET)...${NC}"
	./$(TARGET) $(PARAMS)

$(OBJDIR):
	mkdir $(OBJDIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^
		
$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@rm -f $(TARGET)
	@rm -rf $(OBJECTS)
