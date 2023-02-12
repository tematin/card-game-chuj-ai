class SelectableButtons {
    constructor(min_items, max_items, next_id, submit_callback, change_callback, allowed_cards) {
        this.min_items = min_items;
        this.max_items = max_items;

        this.buttons = $('.selectable');

        if (next_id == '') {
            this.next = null;
        } else {
            this.next = $('#' + next_id);
            this.setup_next_button();
        }

        this.selected_buttons = [];
        this.submit_callback = submit_callback;
        this.change_callback = change_callback;
        this.change_callback([]);

        this.allowed_cards = allowed_cards;

        this.setup_buttons();
    }

    setup_next_button() {
        this.next.on("click", () => {
        let count = this.selected_buttons.length;
            if (count >= this.min_items & count <= this.max_items) {
                this.submit_callback(this.selected_buttons);
            }
        })
    }

    setup_buttons() {
        this.buttons.each((i, button) => {
            let jq_button = $('#' + button.id);
            jq_button.on("click", () => {
                if (this.selected_buttons.includes(button.id)) {
                    this.selected_buttons = this.selected_buttons.filter(item => item !== button.id);
                    jq_button.attr("src", 'images/' + button.id + '.png');
                    this.change_callback(this.selected_buttons);
                } else {
                    if (this.selected_buttons.length >= this.max_items) return;
                    if (this.allowed_cards.length > 0 & !this.allowed_cards.includes(button.id)) return;

                    this.selected_buttons.push(button.id);
                    jq_button.attr("src", 'images/s' + button.id + '.png')
                    this.change_callback(this.selected_buttons);
                }

                if (this.next == null
                    & this.selected_buttons.length <= this.max_items
                    & this.selected_buttons.length >= this.min_items) {
                        this.submit_callback(this.selected_buttons);
                }
            });
        })
    }
}


eel.expose(init_cards_moving_choice)
function init_cards_moving_choice(values) {
    let selectable_buttons = new SelectableButtons(2, 2, "next", (x) => {
        eel.register_action(x);
    },
    (selected_buttons) => {
        moving_card_values(selected_buttons, values);
    }, []);
}

function moving_card_values(selected_buttons, values) {
    fil_values = values.filter(item => selected_buttons.every(x => item[0].includes(x)));

    let possibilities;
    let value_labels = $('.value_labels');
    value_labels.each((i, label) => {
        let card = label.id.replace('num_val_', '');

        if (label.id == 'label_next') {
            possibilities = fil_values.filter(item => item[0].length == selected_buttons.length);
        } else if (selected_buttons.includes(card)) {
            short_selected_buttons = selected_buttons.filter(item => item !== card);
            possibilities = values.filter(item => short_selected_buttons.every(x => item[0].includes(x)));
        } else {
            possibilities = fil_values.filter(item => item[0].includes(card));
        }
        possibilities = possibilities.map(x => x[1]);

        let value;
        if (possibilities.length == 0) {
            value = "";
        } else {
            value = Math.max.apply(Math, possibilities).toFixed(2);
        }
        $('#' + label.id).html(value);
    });
}


eel.expose(init_declaration)
function init_declaration() {
    $('#Declare').on("click", () => {
        eel.register_action("true");
    })

    $('#Pass').on("click", () => {
        eel.register_action("false");
    })
}



eel.expose(init_card_declaration)
function init_card_declaration(values) {
    let selectable_buttons = new SelectableButtons(0, 2, "next", (x) => {
        eel.register_action(x);
    }, (selected_buttons) => {
        moving_card_values(selected_buttons, values);
    }, ['16', '26']);
}


eel.expose(init_regular_play)
function init_regular_play(eligible_actions) {
    let selectable_buttons = new SelectableButtons(1, 1, "", (x) => {
        eel.register_action(x);
    }, () => {}, eligible_actions);
}
