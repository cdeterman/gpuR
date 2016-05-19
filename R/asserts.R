

assert_has_double <- 
    function(plat_idx, device_idx,
             severity = getOption("assertive.severity", "stop"))
    {
        msg <- gettextf(
            "The device %s on platform %s does not support double.
            Try setting type = 'float' or change device if multiple available.",
            get_name_in_parent(device_idx),
            get_name_in_parent(plat_idx),
            domain = "R-assertive.base"
        )
        assert_engine(
            deviceHasDouble,
            plat_idx,
            device_idx,
            msg = msg,
            severity = severity
        )
    }