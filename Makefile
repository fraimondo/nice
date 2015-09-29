.PHONY: all clean wsmi ompk clean-wsmi clean-ompk


WSMI_DIR=nice/wsmi/wsmi_c
OMPK_DIR=nice/algorithms/optimizations/komplexity_c

# all: wsmi ompk
all: ompk

wsmi:
	$(MAKE) -C $(WSMI_DIR)

ompk:
	$(MAKE) -C $(OMPK_DIR)

clean: clean-wsmi clean-ompk

clean-wsmi:
	$(MAKE) -C $(WSMI_DIR) clean

clean-ompk:
	$(MAKE) -C $(OMPK_DIR) clean
