# [開発者より]
#	このMakefileはLinuxでビルドしたいという利用者の要請に応じて作成されましたが、積極的なメンテナンスは期待しないでください。
#	開発の本流はVisual StudioによるWindows環境です。そちらを使うことを強く推奨します。
#	ビルドできない場合などは、自己メンテナンスをお願いします。githubにpull requestを送ってもらえば受け取ります。
#	Thanks to http://urin.github.io/posts/2013/simple-makefile-for-clang/
COMPILER = g++
CFLAGS   = -std=c++11 -Wextra -g -MMD -MP -O3 -fopenmp
LDFLAGS  = -lgomp
LIBS     =
INCLUDE  = -I../../viennacl -I../../boost
TARGET   = ./$(shell basename `readlink -f ..`)
OBJDIR   = ./obj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR = .
endif
SOURCES  = $(wildcard *.cpp)
OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
DEPENDS  = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET)
	@rmdir --ignore-fail-on-non-empty `readlink -f $(OBJDIR)`

-include $(DEPENDS)
