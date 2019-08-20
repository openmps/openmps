COMPILER = clang++
CFLAGS   = -g -MMD -MP -O3 -std=c++14 -stdlib=libc++ -Wall -Wextra -Werror -Wfatal-errors -Wno-unknown-pragmas -march=native -mtune=native -fopenmp
LDFLAGS  = ${CFLAGS} -lomp

LIBS     =
INCLUDE  = -isystem ../../viennacl

TESTDIR = ./
TESTFLAGS= 
TESTLDFLAGS= -lgtest -lgtest_main -L../../googletest/lib

TESTINCLUDE= -isystem ../../googletest/googletest/include
TESTTARGET = ./test_$(shell basename `readlink -f .`)
TESTOBJDIR = ./testobj
ifeq "$(strip $(TESTOBJDIR))" ""
  TESTOBJDIR = .
endif
TESTSOURCES = $(notdir $(wildcard $(TESTDIR)/*.cpp))
TESTOBJECTS  = $(addprefix $(TESTOBJDIR)/, $(TESTSOURCES:.cpp=.o))
TESTDEPENDS = $(TESTOBJECTS:.o=.d)

$(TESTTARGET): $(TESTOBJECTS)
	$(COMPILER) -o $@ $^ $(LDFLAGS) $(TESTLDFLAGS)

$(TESTOBJDIR)/%.o: $(TESTDIR)/%.cpp
	@[ -d $(TESTOBJDIR) ] || mkdir -p $(TESTOBJDIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) $(TESTINCLUDE) $(TESTFLAGS) -o $@ -c $<

all: clean $(TESTTARGET)

clean:
	rm -f $(TESTOBJECTS) $(TESTDEPENDS) $(TESTTARGET)
	@rmdir --ignore-fail-on-non-empty `readlink -f $(TESTOBJDIR)`

-include $(TESTDEPENDS)

