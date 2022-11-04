#ifndef SCU_REG_H
#define SCU_REG_H
#define SCU_BASE 							0x40300000
#define REGISTER_BANK_CLOCK_ENABLE_SET 		0x00000000 	//register bank clock enable set register,p_client_num will decide how many fields will be used!
#define REGISTER_BANK_CLOCK_ENABLE_CLR 		0x00000004 	//register bank clock enable clear register,p_client_num will decide how many fields will be used!
#define REGISTER_BANK_CLOCK_ENABLE_STATUS 	0x00000008 	//register bank Clock Status Register,p_client_num Will Decide How many fields will be used, when any fields is 1,register_bank input clock will be active
#define KERNEL_CLOCK_ENABLE_SET 			0x0000000C 	//Kernel's Kernel_clk clock enable set register,p_client_num will decide how many fields will be used!
#define KERNEL_CLOCK_ENABLE_CLR 			0x00000010 	//Kernel's Kernel_clk clock enable clear register,p_client_num will decide how many fields will be used!
#define KERNEL_CLOCK_ENABLE_STATUS 			0x00000014 	//Kernel's Kernel_clk Clock Status Register,p_client_num Will Decide How many fields will be used, when any fields is 1,register_bank input clock will be active
#define CHIPID 								0x00000018 	//Chip's ID Info,in General only one module need to provide CHIPID for one Chip
#define MODULEID 							0x0000001C 	//moduleID,it will provide below info register interface's Version,Module ID,Module Version
#define CHECKSUM 							0x00000020 	//CHECKSUM is used to indentiy if Register interface setting/parameters are changed or Not,it is caculated from tools ,any source change will cause CHECKSUM value change
#define FUNC_INT_0_ENABLE_SET 				0x00000024 	//Interrupt Enable Set Register
#define FUNC_INT_0_ENABLE_CLEAR 			0x00000028 	//Interrupt Enable Clear Register
#define FUNC_INT_0_ENABLE_STATUS 			0x0000002C 	//Interrupt Enable Status Register
#define FUNC_INT_0_SET 						0x00000030 	//Interrupt Software Trig Register
#define FUNC_INT_0_CLR 						0x00000034 	//Interrupt Software Clear Register
#define FUNC_INT_0_STATUS_ENABLED 			0x00000038 	//enabled Interrupt Status Register
#define FUNC_INT_0_STATUS 					0x0000003C 	//Origin Interrupt Register
#define LMU_BOOT_ADDR_ALIAS_SEL 			0x00000040 	//boot_pin
#define BOOT_STRAP_PIN 						0x00000044 	//boot_strap_pin
#define DDR_INITIAL_DONE 					0x00000048 	//ddr_initial_done
#define UTS_COUNTER_H 						0x0000004C 	//uts_counter_h
#define UTS_COUNTER_L 						0x00000050 	//uts_counter_l
#define UTS_COUNTER_CLR 					0x00000054 	//uts_counter_clr
#define SW_INTERRUPT 						0x00000058 	//sw_int
#define PCIE_INTERRUPT 						0x0000005C 	//pcie_int
#define PCIE_INT_ACK 						0x00000060 	//pcie_int_ack
#define PCIE_USER_LNK_UP 					0x00000064 	//pcie_user_link_up
#define PCIE_MSI_ENABLE 					0x00000068 	//pcie_msi_enable
#define PCIE_MSIX_ENABLE 					0x0000006C 	//pcie_msix_enable
#define MMCM_LOCK 							0x00000440 	//mmcm_lock
#define SW_RST_REQ 							0x00000840 	//sw_rst_req
#define MODULE_RST_CTRL 					0x00000844 	//module_rst_ctrl
#define MODULE_RST_CTRL_MASK 				0x00000848 	//module_rst_ctrl_mask
#define RESET_STATUS 						0x0000084C 	//reset_status
#define WDT_CONTROL 						0x00000C40 	//wdt_control
#define WDT_LOAD_VALUE 						0x00000C44 	//wdt_load_value
#define WDT_INT_LEVEL 						0x00000C48 	//wdt_int_level
#define WDT_RLD_UPD 						0x00000C4C 	//wdt_rld_upd
#define WDT_CNT_VAL 						0x00000C50 	//wdt_cnt_val
#endif
