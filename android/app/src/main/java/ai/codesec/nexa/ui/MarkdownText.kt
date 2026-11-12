package ai.codesec.nexa.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Lightweight, native Jetpack Compose Markdown renderer optimized for offline edge AI chat.
 * Supports bold, italic, inline code, bullet/numbered lists, and syntax-highlighted code blocks with copy.
 */
@Composable
fun MarkdownText(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = MaterialTheme.colorScheme.onSurfaceVariant
) {
    val blocks = remember(text) { parseMarkdownBlocks(text) }

    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        blocks.forEach { block ->
            when (block) {
                is MarkdownBlock.CodeBlock -> {
                    CodeBlockView(
                        language = block.language,
                        code = block.code
                    )
                }
                is MarkdownBlock.BulletItem -> {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(start = 4.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Box(
                            modifier = Modifier
                                .padding(top = 7.dp, end = 8.dp)
                                .size(5.dp)
                                .clip(CircleShape)
                                .background(MaterialTheme.colorScheme.primary)
                        )
                        Text(
                            text = formatInlineMarkdown(block.content, color),
                            style = MaterialTheme.typography.bodyLarge,
                            color = color
                        )
                    }
                }
                is MarkdownBlock.NumberedItem -> {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(start = 4.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Text(
                            text = "${block.number}.",
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(end = 8.dp)
                        )
                        Text(
                            text = formatInlineMarkdown(block.content, color),
                            style = MaterialTheme.typography.bodyLarge,
                            color = color
                        )
                    }
                }
                is MarkdownBlock.Paragraph -> {
                    Text(
                        text = formatInlineMarkdown(block.content, color),
                        style = MaterialTheme.typography.bodyLarge,
                        color = color
                    )
                }
            }
        }
    }
}

@Composable
private fun CodeBlockView(language: String, code: String) {
    val context = LocalContext.current
    var isCopied by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    Surface(
        shape = RoundedCornerShape(8.dp),
        color = Color(0xFF0F172A), // Dark slate code container
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF1E293B))
                    .padding(horizontal = 12.dp, vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = language.ifBlank { "CODE" }.uppercase(),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF94A3B8)
                )

                IconButton(
                    onClick = {
                        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                        val clip = ClipData.newPlainText("code", code)
                        clipboard?.setPrimaryClip(clip)
                        isCopied = true
                        Toast.makeText(context, "Code copied to clipboard", Toast.LENGTH_SHORT).show()
                        scope.launch {
                            delay(2000)
                            isCopied = false
                        }
                    },
                    modifier = Modifier.size(28.dp)
                ) {
                    Icon(
                        imageVector = if (isCopied) Icons.Default.Check else Icons.Default.ContentCopy,
                        contentDescription = "Copy Code",
                        tint = if (isCopied) Color(0xFF10B981) else Color(0xFF94A3B8),
                        modifier = Modifier.size(16.dp)
                    )
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
                    .padding(12.dp)
            ) {
                Text(
                    text = code,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 13.sp,
                    color = Color(0xFFF1F5F9),
                    lineHeight = 18.sp
                )
            }
        }
    }
}

private sealed class MarkdownBlock {
    data class Paragraph(val content: String) : MarkdownBlock()
    data class BulletItem(val content: String) : MarkdownBlock()
    data class NumberedItem(val number: String, val content: String) : MarkdownBlock()
    data class CodeBlock(val language: String, val code: String) : MarkdownBlock()
}

private fun parseMarkdownBlocks(rawText: String): List<MarkdownBlock> {
    val blocks = mutableListOf<MarkdownBlock>()
    val lines = rawText.lines()
    var inCodeBlock = false
    var codeLang = ""
    val codeAccumulator = StringBuilder()
    val paragraphAccumulator = StringBuilder()

    fun flushParagraph() {
        if (paragraphAccumulator.isNotBlank()) {
            blocks.add(MarkdownBlock.Paragraph(paragraphAccumulator.toString().trim()))
            paragraphAccumulator.clear()
        }
    }

    for (line in lines) {
        val trimmed = line.trim()
        if (trimmed.startsWith("```")) {
            if (inCodeBlock) {
                blocks.add(MarkdownBlock.CodeBlock(codeLang, codeAccumulator.toString().trimEnd()))
                codeAccumulator.clear()
                codeLang = ""
                inCodeBlock = false
            } else {
                flushParagraph()
                codeLang = trimmed.removePrefix("```").trim()
                inCodeBlock = true
            }
            continue
        }

        if (inCodeBlock) {
            codeAccumulator.append(line).append("\n")
            continue
        }

        if (trimmed.startsWith("* ") || trimmed.startsWith("- ") || trimmed.startsWith("• ")) {
            flushParagraph()
            val bulletContent = trimmed.substring(2).trim()
            blocks.add(MarkdownBlock.BulletItem(bulletContent))
            continue
        }

        val numberedMatch = Regex("^([0-9]+)\\.\\s+(.*)").find(trimmed)
        if (numberedMatch != null) {
            flushParagraph()
            val num = numberedMatch.groupValues[1]
            val content = numberedMatch.groupValues[2]
            blocks.add(MarkdownBlock.NumberedItem(num, content))
            continue
        }

        if (trimmed.isBlank()) {
            flushParagraph()
        } else {
            if (paragraphAccumulator.isNotEmpty()) {
                paragraphAccumulator.append(" ")
            }
            paragraphAccumulator.append(trimmed)
        }
    }

    if (inCodeBlock) {
        blocks.add(MarkdownBlock.CodeBlock(codeLang, codeAccumulator.toString().trimEnd()))
    }
    flushParagraph()

    return if (blocks.isEmpty()) listOf(MarkdownBlock.Paragraph(rawText)) else blocks
}

private fun formatInlineMarkdown(text: String, defaultColor: Color) = buildAnnotatedString {
    var i = 0
    val n = text.length

    while (i < n) {
        if (text.startsWith("**", i)) {
            val end = text.indexOf("**", i + 2)
            if (end != -1) {
                withStyle(SpanStyle(fontWeight = FontWeight.Bold, color = defaultColor)) {
                    append(text.substring(i + 2, end))
                }
                i = end + 2
                continue
            }
        }

        if (text.startsWith("*", i) && !text.startsWith("**", i)) {
            val end = text.indexOf("*", i + 1)
            if (end != -1) {
                withStyle(SpanStyle(fontStyle = FontStyle.Italic, color = defaultColor)) {
                    append(text.substring(i + 1, end))
                }
                i = end + 1
                continue
            }
        }

        if (text.startsWith("`", i)) {
            val end = text.indexOf("`", i + 1)
            if (end != -1) {
                withStyle(
                    SpanStyle(
                        fontFamily = FontFamily.Monospace,
                        background = Color(0x3364748B),
                        fontWeight = FontWeight.SemiBold
                    )
                ) {
                    append(" ${text.substring(i + 1, end)} ")
                }
                i = end + 1
                continue
            }
        }

        append(text[i])
        i++
    }
}
