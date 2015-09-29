.PHONY: all clean jivaro ompk clean-jivaro clean-ompk

JIVARO_DIR=nice/algorithms/optimizations/jivaro
OMPK_DIR=nice/algorithms/optimizations/ompk

all: jivaro ompk

jivaro:
	$(MAKE) -C $(JIVARO_DIR)

ompk:
	$(MAKE) -C $(OMPK_DIR)

clean: clean-jivaro clean-ompk

clean-jivaro:
	$(MAKE) -C $(JIVARO_DIR) clean

clean-ompk:
	$(MAKE) -C $(OMPK_DIR) clean
