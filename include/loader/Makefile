CPPFLAGS = -I../

.PHONY: clean

.$(R_ARCH)/libOpenCL.a: .$(R_ARCH)/icd.o .$(R_ARCH)/icd_dispatch.o .$(R_ARCH)/$(ICD_OS).o
	$(RM) $@
	$(AR) rcs $@ $^

.$(R_ARCH)/icd.o: icd.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
    
.$(R_ARCH)/icd_dispatch.o: icd_dispatch.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
    
.$(R_ARCH)/$(ICD_OS).o: $(ICD_OS).c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean:
	$(RM) .$(R_ARCH)/*.o
	$(RM) .$(R_ARCH)/*.a
