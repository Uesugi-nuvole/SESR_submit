# xsct generate_platform.tcl <new workspace> <XSA>
if {$argc != 2} { error "Expected workspace and XSA arguments" }
setws [file normalize [lindex $argv 0]]
platform create -name completion_platform -hw [file normalize [lindex $argv 1]] -proc microblaze_0 -os standalone
platform generate
puts "PLATFORM_GENERATED"
