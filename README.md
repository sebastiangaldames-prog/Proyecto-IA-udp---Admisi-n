# 🎓 Sistema Predictivo de Admisión Universitaria (Control #03 - IA Aplicada UDP)

Repositorio oficial del grupo de trabajo para el desarrollo del proyecto semestral de Inteligencia Artificial.

## 👥 Integrantes
* Sebastián Galdames
* Nicolás Rodríguez
* Constanza Toro

## 📌 Descripción del Proyecto
Solución de IA basada en Aprendizaje Supervisado que procesa microdatos históricos oficiales (DEMRE y MINEDUC 2024-2026) para estimar la viabilidad y probabilidad porcentual de ingreso de un postulante a carreras universitarias en Chile. Incluye un flujo conversacional inteligente de 10 pasos.

## ⚙️ Componentes Principales
1. **Consolidación y Cruce de Datos:** Integración automatizada de fuentes estatales mediante llaves anonimizadas.
2. **Evaluación Comparativa de Modelos:** Entrenamiento y contraste de 3 algoritmos (`Naive Bayes`, `Árbol de Decisión` y `Random Forest`), seleccionando el de mayor `F1-Score`.
3. **Asistente Conversacional (Gradio):** Interfaz interactiva guiada para la simulación vocacional en tiempo real.
