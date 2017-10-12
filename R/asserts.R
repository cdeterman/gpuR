
#' @title Does device have 'double' support?
#' @description Function to query if device (identified by index) supports
#' double precision
#' @param device_idx An integer indicating which device to query
#' @param context_idx An integer indicating which context to query
#' @param severity How severe should the consequences of the assertion be?
#' @return Returns nothing but throws an error if device does not support
#' double precision
#' @seealso \link{deviceHasDouble}
#' @author Charles Determan Jr.
#' @export
assert_has_double <- 
    function(device_idx, context_idx,
             severity = getOption("assertive.severity", "stop"))
    {
        msg <- gettextf(
            "The device %s on context %s does not support double.
            Try setting type = 'float' or change device if multiple available.",
            get_name_in_parent(device_idx),
            get_name_in_parent(context_idx),
            domain = "R-assertive.base"
        )
        assert_engine(
            deviceHasDouble,
            device_idx,
            context_idx,
            msg = msg,
            severity = severity
        )
    }
