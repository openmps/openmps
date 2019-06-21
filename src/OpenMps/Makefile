# [開発者より]
#	このMakefileはLinuxでビルドしたいという利用者の要請に応じて作成されましたが、積極的なメンテナンスは期待しないでください。
#	開発の本流はVisual StudioによるWindows環境です。そちらを使うことを強く推奨します。
#	ビルドできない場合などは、自己メンテナンスをお願いします。githubにpull requestを送ってもらえば受け取ります。
#	Thanks to http://urin.github.io/posts/2013/simple-makefile-for-clang/
COMPILER = clang++
CFLAGS   = -g -MMD -MP -O3 -std=c++14 -stdlib=libc++ -Wall -Wextra -Werror -Wfatal-errors -Wno-unknown-pragmas -march=native -mtune=native -fopenmp
LDFLAGS  = ${CFLAGS} -lomp

LIBS     =
INCLUDE  = -isystem ../viennacl 
TARGET   = ./$(shell basename `readlink -f .`)
OBJDIR   = ./obj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR = .
endif
SOURCES  = $(wildcard *.cpp)
OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
DEPENDS  = $(OBJECTS:.o=.d)

TESTDIR = ./test
TESTFLAGS= 
TESTLDFLAGS= -lgtest -lgtest_main -L../googletest/lib

TESTINCLUDE= -isystem ../googletest/googletest/include
TESTTARGET = ./test_$(shell basename `readlink -f .`)
TESTOBJDIR = ./testobj
ifeq "$(strip $(TESTOBJDIR))" ""
  TESTOBJDIR = .
endif
TESTSOURCES = $(notdir $(wildcard $(TESTDIR)/*.cpp))
TESTOBJECTS  = $(addprefix $(TESTOBJDIR)/, $(TESTSOURCES:.cpp=.o))
TESTDEPENDS = $(TESTOBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

$(TESTTARGET): $(TESTOBJECTS)
	$(COMPILER) -o $@ $^ $(LDFLAGS) $(TESTLDFLAGS)

$(TESTOBJDIR)/%.o: $(TESTDIR)/%.cpp
	@[ -d $(TESTOBJDIR) ] || mkdir -p $(TESTOBJDIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) $(TESTINCLUDE) $(TESTFLAGS) -o $@ -c $<

all: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET)
	rm -f $(TESTOBJECTS) $(TESTDEPENDS) $(TESTTARGET)
	@rmdir --ignore-fail-on-non-empty `readlink -f $(OBJDIR)`
	@rmdir --ignore-fail-on-non-empty `readlink -f $(TESTOBJDIR)`

test: $(TESTTARGET)

-include $(DEPENDS)
-include $(TESTDEPENDS)

