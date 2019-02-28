CXXFLAGS  = -Ofast -Wall -std=c++17 -march=native -lpthread -ltensorflow

TARGET    = needletail
SRCS      = $(shell ls *.cpp)
OBJS      = $(SRCS:%.cpp=%.o)
DEPS      = $(SRCS:%.cpp=%.d)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS)

-include $(DEPS)

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -MMD -MP

clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS)
