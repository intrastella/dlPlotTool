if(!window.dash_clientside) {window.dash_clientside = {};}

window.dash_clientside.clientside = {

  stickyHeader: function(id) {
    var header = document.getElementById("app-page-header");
    var sticky = header.offsetTop;
    var descript = document.getElementById("dscript");

    window.onscroll = function() {
      if (window.pageYOffset > sticky) {
        header.classList.add("sticky");
        descript.classList.add("sticky");
      } else {
        descript.classList.remove("sticky");
        header.classList.remove("sticky");
      }
    };

    return window.dash_clientside.no_update
  },
}

