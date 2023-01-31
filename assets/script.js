if(!window.dash_clientside) {window.dash_clientside = {};}

window.dash_clientside.clientside = {

  stickyHeader: function(id) {
    var header = document.getElementById("app-page-header");
    var sticky = header.offsetTop;
    var descript = document.getElementById("dscript");
    var logo = document.getElementById("logo");
    var in_logo = document.getElementById("in-logo");
    var nav = document.getElementById("navcontainer");

    window.onscroll = function() {
      if (window.pageYOffset > sticky) {
        header.classList.add("sticky");
        descript.classList.add("sticky");
        in_logo.classList.add("sticky");
        logo.classList.add("sticky");
        nav.classList.add("sticky");
      } else {
        descript.classList.remove("sticky");
        header.classList.remove("sticky");
        in_logo.classList.remove("sticky");
        logo.classList.remove("sticky");
        nav.classList.remove("sticky");
      }
    };
    return window.dash_clientside.no_update
  },
}

