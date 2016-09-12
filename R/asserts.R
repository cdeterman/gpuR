
#' @title Does device have 'double' support?
#' @description Function to query if device (identified by index) supports
#' double precision
#' @param plat_idx An integer indicating which platform to query
#' @param device_idx An integer indicating which device to query
#' @param severity How severe should the consequences of the assertion be?
#' @return Returns nothing but throws an error if device does not support
#' double precision
#' @seealso \link{deviceHasDouble}
#' @author Charles Determan Jr.
#' @export
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
