<script lang="ts">
	export let ctrlNumMotors = 4
	export let ctrlLevelsPerMotor = 16
	export let ctrlStateNeurons = 4
	export let ctrlStateBits = 24
	export let ctrlOutputBits = 24
	export let ctrlInputWindowK = 4
	export let ctrlBitsPerFeature = 8
	export let ctrlEvalEpisodes = 20
	export let ctrlSteps = 1500
	export let ctrlTiltDeg = 15.0
	export let ctrlDeltaControl = false
	export let ctrlSeed = 0
</script>

<div class="form-section">
	<h2>Controller Configuration</h2>
	<p class="section-hint">WNN drone attitude controller (custom sim). Metrics are reward / stable % / attitude-error°, not F1/CE.</p>
	<div class="form-row">
		<div class="form-group">
			<label for="ctrlNumMotors">Num Motors</label>
			<input type="number" id="ctrlNumMotors" bind:value={ctrlNumMotors} min="1" max="8" />
			<span class="field-hint">Quadcopter = 4</span>
		</div>
		<div class="form-group">
			<label for="ctrlLevelsPerMotor">Levels / Motor</label>
			<input type="number" id="ctrlLevelsPerMotor" bind:value={ctrlLevelsPerMotor} min="2" max="2048" />
			<span class="field-hint">PWM thermometer levels (16 default, upgradeable to 256/2048)</span>
		</div>
	</div>
	<div class="form-row">
		<div class="form-group">
			<label for="ctrlStateNeurons">State Neurons</label>
			<input type="number" id="ctrlStateNeurons" bind:value={ctrlStateNeurons} min="1" max="64" />
			<span class="field-hint">Recurrent state-layer neurons (held-out run used 4)</span>
		</div>
		<div class="form-group">
			<label for="ctrlInputWindowK">Input Window K</label>
			<input type="number" id="ctrlInputWindowK" bind:value={ctrlInputWindowK} min="1" max="16" />
			<span class="field-hint">Sensor frames in the input window</span>
		</div>
	</div>
	<div class="form-row">
		<div class="form-group">
			<label for="ctrlStateBits">State Bits / Neuron</label>
			<input type="number" id="ctrlStateBits" bind:value={ctrlStateBits} min="1" max="64" />
			<span class="field-hint">Floor: 2×state_neurons+1 (forced full-state prefix)</span>
		</div>
		<div class="form-group">
			<label for="ctrlOutputBits">Output Bits / Neuron</label>
			<input type="number" id="ctrlOutputBits" bind:value={ctrlOutputBits} min="1" max="64" />
		</div>
		<div class="form-group">
			<label for="ctrlBitsPerFeature">Bits / Feature</label>
			<input type="number" id="ctrlBitsPerFeature" bind:value={ctrlBitsPerFeature} min="2" max="16" />
			<span class="field-hint">Thermometer bits per IMU feature</span>
		</div>
	</div>
	<h3>Episode / Evaluation</h3>
	<div class="form-row">
		<div class="form-group">
			<label for="ctrlEvalEpisodes">Eval Episodes</label>
			<input type="number" id="ctrlEvalEpisodes" bind:value={ctrlEvalEpisodes} min="1" max="200" />
			<span class="field-hint">Closed-loop episodes averaged per genome</span>
		</div>
		<div class="form-group">
			<label for="ctrlSteps">Steps / Episode</label>
			<input type="number" id="ctrlSteps" bind:value={ctrlSteps} min="100" max="10000" step="100" />
			<span class="field-hint">Sim steps (dt=0.001 → 1500 = 1.5s)</span>
		</div>
		<div class="form-group">
			<label for="ctrlTiltDeg">Max Initial Tilt (°)</label>
			<input type="number" id="ctrlTiltDeg" bind:value={ctrlTiltDeg} min="1" max="90" step="1" />
			<span class="field-hint">Initial-condition disturbance range</span>
		</div>
	</div>
	<div class="form-row">
		<div class="form-group">
			<label for="ctrlSeed">Seed</label>
			<input type="number" id="ctrlSeed" bind:value={ctrlSeed} min="0" />
			<span class="field-hint">Evolution / IC seed</span>
		</div>
		<div class="form-group">
			<label for="ctrlDeltaControl">
				<input type="checkbox" id="ctrlDeltaControl" bind:checked={ctrlDeltaControl} />
				Delta Control
			</label>
			<span class="field-hint">Output = Δthrottle instead of absolute</span>
		</div>
	</div>
</div>

<style>
  .form-section {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0; /* inside .form-columns the page zeroed this */
    box-shadow: var(--glass-shadow), var(--glass-inset);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
  }

  .form-section:hover {
    box-shadow: var(--glass-shadow-hover), var(--glass-inset);
    border-color: var(--glass-border-highlight);
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 0.75rem 0;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .section-hint {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: -0.5rem 0 0.75rem 0;
  }

  .form-group {
    margin-bottom: 0.75rem;
  }

  .form-group:last-child {
    margin-bottom: 0;
  }

  .field-hint {
    display: block;
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }

  input[type="checkbox"] {
    width: auto;
    margin-right: 0.5rem;
    vertical-align: middle;
  }

  label:has(input[type="checkbox"]) {
    display: flex;
    align-items: center;
    cursor: pointer;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }

  label {
    display: block;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 0.375rem;
  }

  input {
    width: 100%;
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    background: var(--glass-input-bg);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: var(--text-primary);
    font-size: 1rem;
    font-family: inherit;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }

  input:focus {
    outline: none;
    border-color: rgba(59, 130, 246, 0.6);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15), 0 0 0 3px rgba(59, 130, 246, 0.15);
  }
</style>
