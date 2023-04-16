class SelectableButtons {
    constructor(options) {
        this.min_items = options['min_items'];
        this.max_items = options['max_items'];

        if (options['grayscale'] ?? false) {
            this.unselected_name = 'g';
            this.selected_name = '';
        } else {
            this.unselected_name = '';
            this.selected_name = 's';
        }

        this.buttons = $('.selectable');

        this.next = options['next_id'] ?? null;
        if (this.next !== null) {
            this.next = $('#' + this.next);
            this.setup_next_button();
        }

        this.selected_buttons = [];
        this.submit_callback = options['submit_callback'] ?? ((x) => {eel.register_action(x)});
        this.change_callback = options['change_callback'] ?? (() => {});
        this.change_callback([]);

        this.allowed_cards = options['allowed_cards'] ?? [];

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
        eel.log(this.unselected_name);
        this.buttons.each((i, button) => {
            let jq_button = $('#' + button.id);
            jq_button.attr("src", 'images/' + this.unselected_name + button.id + '.png');
            jq_button.on("click", () => {
                if (this.selected_buttons.includes(button.id)) {
                    this.selected_buttons = this.selected_buttons.filter(item => item !== button.id);
                    jq_button.attr("src", 'images/' + this.unselected_name + button.id + '.png');
                    this.change_callback(this.selected_buttons);
                } else {
                    if (this.selected_buttons.length >= this.max_items) return;
                    if (this.allowed_cards.length > 0 & !this.allowed_cards.includes(button.id)) return;

                    this.selected_buttons.push(button.id);
                    jq_button.attr("src", 'images/' + this.selected_name + button.id + '.png');
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

    let selectable_buttons = new SelectableButtons({
        'min_items': 2,
        'max_items': 2,
        'next_id': 'next',
        'change_callback': (selected_buttons) => {
            moving_card_values(selected_buttons, values);
        }
    });
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
    let selectable_buttons = new SelectableButtons({
        'min_items': 0,
        'max_items': 2,
        'next_id': 'next',
        'change_callback': (selected_buttons) => {
            moving_card_values(selected_buttons, values);
        },
        'allowed_cards': ['16', '26']
    });
}


eel.expose(init_regular_play)
function init_regular_play(eligible_actions) {
    let selectable_buttons = new SelectableButtons({
        'min_items': 1,
        'max_items': 1,
        'allowed_cards': eligible_actions
    });
}


eel.expose(init_starting_hand)
function init_starting_hand(values) {
    let selectable_buttons = new SelectableButtons({
        'min_items': 12,
        'max_items': 12,
        'next_id': 'next',
        'change_callback': update_value,
        'grayscale': true
    });
}


async function update_value(selected_buttons) {
    if (selected_buttons.length != 12) {
        $('#value_label').html('');
        return;
    }
    let x = await eel.evaluate_hand(selected_buttons)();
    $('#value_label').html(x);
}
