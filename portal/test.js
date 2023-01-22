eel.expose(update_html);
function update_html(identifier, value) {
    $("#" + identifier).html(value);
}


eel.expose(register_buttons);
function register_buttons(buttons) {
    for (let button of buttons) {
        $('#' + button).on("click", () => {
            eel.button_clicked(button);
        })
    }
}
