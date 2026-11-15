"""Generate ``examples/manual-colaborador.pdf``, a fictional handbook used in the demo.

Requires reportlab (``pip install -e ".[dev]"``). Run from the repository root:

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

SECTIONS = [
    (
        "1. Boas-vindas e rotina de trabalho",
        [
            "A Acme Tecnologia é uma empresa fictícia criada apenas para demonstrar o projeto "
            "rag-documentos. Este manual reúne as principais políticas internas para novos "
            "colaboradores.",
            "A jornada padrão é de 40 horas semanais, distribuídas de segunda a sexta-feira. "
            "O horário de entrada é flexível, entre 8h e 10h, desde que a jornada diária de "
            "8 horas seja cumprida. As reuniões do time acontecem preferencialmente entre 10h "
            "e 16h, para que todos estejam disponíveis.",
            "Os canais oficiais de comunicação são o Slack, para conversas do dia a dia, e o "
            "e-mail corporativo, para comunicados formais e assuntos com clientes externos. "
            "Chamados de suporte de TI devem ser abertos no portal de atendimento, nunca por "
            "mensagem direta.",
            "Todo novo colaborador passa por um período de integração de 30 dias, com um padrinho "
            "ou madrinha designado pela liderança. Ao final do período, há uma conversa de "
            "alinhamento sobre expectativas e próximos passos.",
        ],
    ),
    (
        "2. Férias e licenças",
        [
            "Após 12 meses de trabalho, o colaborador tem direito a 30 dias corridos de férias. "
            "As férias podem ser divididas em até três períodos, sendo que um deles deve ter no "
            "mínimo 14 dias corridos e os demais, no mínimo 5 dias corridos cada.",
            "A solicitação deve ser feita pelo portal do colaborador com pelo menos 45 dias de "
            "antecedência e depende da aprovação da liderança imediata. Em períodos de fechamento "
            "trimestral, a aprovação pode ser adiada por até duas semanas.",
            "A licença-maternidade é de 180 dias. A licença-paternidade é de 20 dias corridos, "
            "contados a partir do nascimento ou da adoção. Em caso de casamento, o colaborador "
            "tem 3 dias de licença e, em caso de falecimento de familiar próximo, 2 dias úteis.",
            "Atestados médicos devem ser enviados ao RH em até 48 horas após o retorno "
            "ao trabalho.",
        ],
    ),
    (
        "3. Trabalho remoto e equipamentos",
        [
            "A Acme adota o modelo híbrido: dois dias presenciais por semana, às terças e "
            "quintas-feiras, e três dias de trabalho remoto. Times que atendem clientes "
            "presencialmente podem ter escalas diferentes, definidas pela liderança.",
            "Para custear despesas de casa, o colaborador recebe um auxílio internet de R$ 100 "
            "por mês, pago junto com a folha de pagamento. A empresa fornece notebook e monitor, "
            "que devem ser devolvidos no desligamento.",
            "Por segurança, o acesso a sistemas internos fora do escritório exige o uso "
            "obrigatório da VPN corporativa. Redes Wi-Fi públicas não devem ser usadas sem a VPN "
            "ativa.",
            "É permitido trabalhar de outro país por até 15 dias por ano, mediante aprovação "
            "prévia da liderança e do RH.",
        ],
    ),
    (
        "4. Benefícios e reembolsos",
        [
            "O vale-refeição é de R$ 45 por dia útil e pode ser usado em restaurantes e "
            "supermercados. O plano de saúde tem cobertura nacional, com coparticipação em "
            "consultas e exames, e pode ser estendido a dependentes.",
            "O auxílio-educação cobre até 70% da mensalidade de cursos de graduação, "
            "pós-graduação e idiomas, com limite de R$ 800 por mês. É necessário apresentar "
            "comprovante de matrícula a cada semestre.",
            "Despesas de trabalho, como viagens a serviço, devem ser reembolsadas mediante nota "
            "fiscal. O pedido deve ser feito em até 60 dias após a despesa, e o pagamento ocorre "
            "em até 30 dias após a aprovação. Para hospedagem, o limite é de R$ 450 por diária.",
        ],
    ),
    (
        "5. Segurança da informação e conduta",
        [
            "Senhas devem ter no mínimo 14 caracteres e a autenticação em dois fatores é "
            "obrigatória em todos os sistemas corporativos. Nunca compartilhe credenciais, nem "
            "mesmo com colegas de equipe.",
            "Qualquer incidente de segurança, como suspeita de phishing ou perda de equipamento, "
            "deve ser reportado em até 24 horas ao time de segurança pelo e-mail "
            "seguranca@acme.example.",
            "Presentes de fornecedores ou clientes podem ser aceitos apenas até o valor de "
            "R$ 150 e devem ser declarados no portal de conformidade. O canal de ética é anônimo "
            "e está disponível 24 horas por dia.",
        ],
    ),
]


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        2 * cm, 1.2 * cm, "Acme Tecnologia (documento fictício) - Manual do Colaborador"
    )
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.restoreState()


def build(output: Path) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=11, leading=16,
                          alignment=TA_JUSTIFY, spaceAfter=10)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                            title="Manual do Colaborador - Acme Tecnologia")
    story = []
    for position, (title, paragraphs) in enumerate(SECTIONS):
        if position:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 0.4 * cm))
        story.extend(Paragraph(text, body) for text in paragraphs)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "examples" / "manual-colaborador.pdf"
    build(target)
    print(f"Gerado: {target}")
