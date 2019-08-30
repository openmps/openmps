COMPILER = clang++
CFLAGS   = -g -MMD -MP -O3 -std=c++14 -stdlib=libc++ -Wall -Wextra -Werror -Wfatal-errors -Wno-unknown-pragmas -march=native -mtune=native -fopenmp
LDFLAGS= ${CFLAGS} -lomp -lgtest -lgtest_main -L../../googletest/lib
INCLUDE  = -isystem ../../viennacl -isystem ../../googletest/googletest/include

TESTDIR = ./
TARGET = ./test_$(shell basename `readlink -f ..`)
OBJDIR = ./testobj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR = .
endif
SOURCES = $(notdir $(wildcard $(TESTDIR)/*.cpp))
OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
DEPENDS = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS)
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(TESTDIR)/%.cpp
	@[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET)
	@rmdir --ignore-fail-on-non-empty `readlink -f $(OBJDIR)`

-include $(DEPENDS)

