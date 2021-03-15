/**
 * jspsych-survey-html-form
 * a jspsych plugin for free html forms
 *
 * Jan Simson
 *
 * documentation: docs.jspsych.org
 *
 */

jsPsych.plugins['survey-html-form-slider'] = (function() {

  var plugin = {};

  // hacky, but whatever [javascript...] :)
  var slider_1_moved = false;
  var slider_2_moved = false;
  var slider_3_moved = false;
  var slider_4_moved = false;
  var slider_5_moved = false;
  var slider_6_moved = false;

  plugin.info = {
    name: 'survey-html-form-slider',
    description: '',
    parameters: {
      html: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'HTML',
        default: null,
        description: 'HTML formatted string containing all the input elements to display. Every element has to have its own distinctive name attribute. The <form> tag must not be included and is generated by the plugin.'
      },
      preamble: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Preamble',
        default: null,
        description: 'HTML formatted string to display at the top of the page above all the questions.'
      },
      button_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'The text that appears on the button to finish the trial.'
      },
      dataAsArray: {
        type: jsPsych.plugins.parameterType.BOOLEAN,
        pretty_name: 'Data As Array',
        default:  false,
        description: 'Retrieve the data as an array e.g. [{name: "INPUT_NAME", value: "INPUT_VALUE"}, ...] instead of an object e.g. {INPUT_NAME: INPUT_VALUE, ...}.'
      },
      require_movement: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Require movement',
        default: false,
        description: 'If true, the participant will have to move the sliders before continuing.'
      },
      items: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Number of sliders',
        default: '',
        description: 'Number of sliders to show in the form.'
      }
    }
  }

  plugin.trial = function(display_element, trial) {
    // array to store flags of movement of sliders
    // hacky, but whatever [javascript...] :)
    if (trial.require_movement){
    var sliders_moved = [];
      for(var i = 0; i < trial.items; i++) {
        sliders_moved[i] = 0;
      }
    }

    var html = '';
    // show preamble text
    if(trial.preamble !== null){
      html += '<div id="jspsych-survey-html-form-preamble" class="jspsych-survey-html-form-preamble">'+trial.preamble+'</div>';
    }
    // start form
    html += '<form id="jspsych-survey-html-form">'

    // add form HTML / input elements
    html += trial.html;

    // add submit button
    html += '<input type="submit" id="jspsych-survey-html-form-next" class="jspsych-btn jspsych-survey-html-form" '+ (trial.require_movement ? "disabled" : "") + ' value="'+trial.button_label+'"></input>';

    html += '</form>'
    display_element.innerHTML = html;

    // check if all entries in row are unique
    if(trial.require_movement){
      for (var item = 0; item < trial.items; item++) { // iterate over rows
        display_element.querySelector('#slider-' + item).addEventListener('change', function(){            
          var numberPattern = /\d+/g;
          numbers = this.id.match(numberPattern);
          var item_event = numbers[0];  // extract item from id
          // update flag in the array
          sliders_moved[item_event] = 1;
          // get number of sliders moved
          var sliders_moved_count = 0;
          for (var i = 0; i < trial.items; i++) {
             if (sliders_moved[i]) {
                sliders_moved_count++;
             }
          }
          // update status of button
          if (sliders_moved_count === trial.items) {  // check if all sliders were moved
            display_element.querySelector('#jspsych-survey-html-form-next').disabled = false;
          }
        });
      }
    }

    display_element.querySelector('#jspsych-survey-html-form').addEventListener('submit', function(event) {
      // don't submit form
      event.preventDefault();

      // measure response time
      var endTime = performance.now();
      var response_time = endTime - startTime;

      var question_data = serializeArray(this);

      if (!trial.dataAsArray) {
        question_data = objectifyForm(question_data);
      }

      // save data
      var trialdata = {
        "rt": response_time,
        "responses": JSON.stringify(question_data)
      };

      display_element.innerHTML = '';

      // next trial
      jsPsych.finishTrial(trialdata);
    });

    var startTime = performance.now();
  };

  /*!
   * Serialize all form data into an array
   * (c) 2018 Chris Ferdinandi, MIT License, https://gomakethings.com
   * @param  {Node}   form The form to serialize
   * @return {String}      The serialized form data
   */
  var serializeArray = function (form) {
    // Setup our serialized data
    var serialized = [];

    // Loop through each field in the form
    for (var i = 0; i < form.elements.length; i++) {
      var field = form.elements[i];

      // Don't serialize fields without a name, submits, buttons, file and reset inputs, and disabled fields
      if (!field.name || field.disabled || field.type === 'file' || field.type === 'reset' || field.type === 'submit' || field.type === 'button') continue;

      // If a multi-select, get all selections
      if (field.type === 'select-multiple') {
        for (var n = 0; n < field.options.length; n++) {
          if (!field.options[n].selected) continue;
          serialized.push({
            name: field.name,
            value: field.options[n].value
          });
        }
      }

      // Convert field data to a query string
      else if ((field.type !== 'checkbox' && field.type !== 'radio') || field.checked) {
        serialized.push({
          name: field.name,
          value: field.value
        });
      }
    }

    return serialized;
  };

  // from https://stackoverflow.com/questions/1184624/convert-form-data-to-javascript-object-with-jquery
  function objectifyForm(formArray) {//serialize data function
    var returnArray = {};
    for (var i = 0; i < formArray.length; i++){
      returnArray[formArray[i]['name']] = formArray[i]['value'];
    }
    return returnArray;
  }

  return plugin;
})();