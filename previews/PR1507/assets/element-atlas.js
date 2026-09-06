// Progressive enhancement: without JavaScript, every basis remains visible.
// No plot library, remote requests, or client-side numerical approximation.
function initializeElementAtlas() {
    for (const explorer of document.querySelectorAll(".atlas-explorer")) {
        if (explorer.dataset.initialized) continue;
        explorer.dataset.initialized = "true";
        const select = explorer.querySelector("select");
        const panels = [...explorer.querySelectorAll(".atlas-basis")];
        const dofs = explorer.querySelector(".atlas-dof-controls");
        const update = () => {
            const previous = panels.find(panel => !panel.hidden);
            const scrollLeft = previous?.querySelector("figure").scrollLeft || 0;
            for (const panel of panels) panel.hidden = panel.dataset.basis !== select.value;
            const active = panels.find(panel => !panel.hidden);
            active.querySelector(".atlas-diagram").append(dofs);
            for (const button of dofs.querySelectorAll("button")) {
                button.setAttribute("aria-pressed", String(button.dataset.dof === select.value));
            }
            active.querySelector("figure").scrollLeft = scrollLeft;
            explorer.querySelector(".atlas-counter").textContent = `${select.value} / ${panels.length}`;
            return active;
        };
        select.addEventListener("change", update);
        explorer.addEventListener("click", event => {
            const button = event.target.closest(".atlas-dof-button");
            if (!button) return;
            select.value = button.dataset.dof;
            const active = update();
            // The previous panel is now hidden. Keep keyboard focus on the same
            // dof in its replacement, without jumping a scrolled mobile figure.
            active.querySelector(`[data-dof="${select.value}"]`).focus({ preventScroll: true });
        });
        explorer.querySelector(".atlas-controls").hidden = false;
        update();
    }
}
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeElementAtlas);
} else {
    initializeElementAtlas();
}
